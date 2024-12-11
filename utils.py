import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import zoom


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        
        smooth = 1e-5
        intersect = torch.sum(score * target)
        
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        
        class_wise_dice = []
        loss = 0.0
        
        # Background(index 0) 제외, Foreground(나머지 클래스)에 대해서만 Dice Loss 계산
        for i in range(1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        # Background를 제외한 클래스 수로 나눠서 평균 계산
        return loss / (self.n_classes - 1)
    
    
def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Soerensen-Dice coefficient."""
    volume_sum = mask_gt.sum() + mask_pred.sum()
    
    if volume_sum == 0:
        return np.NaN
    
    volume_intersect = (mask_gt & mask_pred).sum()
    
    return 2 * volume_intersect / volume_sum


def compute_average_precision(mask_gt, mask_pred):
    """Compute Average Precision (AP) score."""
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    
    return auc(recall, precision)


def compute_hausdorff_distance(mask_gt, mask_pred):
    """Compute Hausdorff Distance (HD)."""
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    
    hd_1 = directed_hausdorff(gt_points, pred_points)[0]
    hd_2 = directed_hausdorff(pred_points, gt_points)[0]
    
    return max(hd_1, hd_2)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if gt.sum() == 0 and pred.sum() == 0:
        dice = 1
        m_ap = 1
        hd = 0
    elif gt.sum() == 0 and pred.sum() > 0:
        dice = 0
        m_ap = 0
        hd = np.NaN
    else:
        dice = compute_dice_coefficient(gt, pred)
        m_ap = compute_average_precision(gt, pred)
        hd = compute_hausdorff_distance(gt, pred)

    return dice, m_ap, hd


def test_single_volume(image_volume, prev_volume, next_volume, label_volume, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1):
    h, w = patch_size  # 학습에 사용한 (224, 224) 크기
    original_h, original_w = image_volume.shape[1], image_volume.shape[2]  # 원본 크기 (512, 512)
    depth = image_volume.shape[0]  # 슬라이스 개수

    # 예측 및 레이블 볼륨 초기화 (원본 크기인 512x512로 설정)
    prediction_volume = np.zeros((depth, original_h, original_w), dtype=np.float32)
    label_volume_resized = np.zeros((depth, original_h, original_w), dtype=np.float32)

    for slice_index in range(depth):
        # 슬라이스 데이터 가져오기
        slice_img = image_volume[slice_index].cpu().detach().numpy()
        slice_prev = prev_volume[slice_index].cpu().detach().numpy()
        slice_next = next_volume[slice_index].cpu().detach().numpy()
        slice_label = label_volume[slice_index].cpu().detach().numpy()

        # 슬라이스를 모델 입력 크기인 (224, 224)로 조정
        slice_img_resized = zoom(slice_img, (h / original_h, w / original_w), order=3)
        slice_prev_resized = zoom(slice_prev, (h / original_h, w / original_w), order=3)
        slice_next_resized = zoom(slice_next, (h / original_h, w / original_w), order=3)
        slice_label_resized = zoom(slice_label, (h / original_h, w / original_w), order=0)

        # PyTorch 텐서로 변환하고 GPU로 이동
        input_img = torch.from_numpy(slice_img_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        input_prev = torch.from_numpy(slice_prev_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        input_next = torch.from_numpy(slice_next_resized).unsqueeze(0).unsqueeze(0).float().cuda()

        # 모델 추론
        net.eval()
        with torch.no_grad():
            slice_inputs = (input_prev, input_img, input_next)
            outputs, attn_maps, _ = net(slice_inputs, return_attn=True)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            # 예측 결과를 원본 크기 (512, 512)로 복원
            pred = zoom(out, (original_h / h, original_w / w), order=0)

            # 예측 및 레이블 결과 저장
            prediction_volume[slice_index] = pred
            label_volume_resized[slice_index] = slice_label

    # 메트릭 계산
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction_volume == i, label_volume_resized == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image_volume.cpu().detach().numpy())
        prd_itk = sitk.GetImageFromArray(prediction_volume)
        lab_itk = sitk.GetImageFromArray(label_volume.cpu().detach().numpy())
        img_itk.SetSpacing((0.375, 0.375, z_spacing))
        prd_itk.SetSpacing((0.375, 0.375, z_spacing))
        lab_itk.SetSpacing((0.375, 0.375, z_spacing))
        sitk.WriteImage(prd_itk, f"{test_save_path}/{case}_pred.nii.gz")
        sitk.WriteImage(img_itk, f"{test_save_path}/{case}_img.nii.gz")
        sitk.WriteImage(lab_itk, f"{test_save_path}/{case}_gt.nii.gz")

    return metric_list