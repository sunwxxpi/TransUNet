import os
import sys
import random
import argparse
import logging
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.dataset import COCA_dataset
from utils import test_single_volume
from networks.transunet.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, default='./data/COCA/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str, default='COCA', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./data/COCA/lists_COCA', help='list dir')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=96, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
args = parser.parse_args()

def inference(args, model, test_save_path=None):
    db_test = COCA_dataset(
        base_dir=args.volume_path, 
        split="test", 
        list_dir=args.list_dir,
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    
    metric_list_all = []
    
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image_volume = sampled_batch["image"].squeeze(0)
        prev_volume = sampled_batch["prev_image"].squeeze(0)
        next_volume = sampled_batch["next_image"].squeeze(0)
        label_volume = sampled_batch["label"].squeeze(0)
        case_name = sampled_batch['case_name'][0]
        
        # 메트릭 계산 및 로깅
        metric_i = test_single_volume(image_volume, prev_volume, next_volume, label_volume,
                                      model, classes=args.num_classes, patch_size=[args.img_size, args.img_size], 
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list_all.append(metric_i)
        
        mean_dice_case = np.nanmean(metric_i, axis=0)[0]
        mean_m_ap_case = np.nanmean(metric_i, axis=0)[1]
        mean_hd95_case = np.nanmean(metric_i, axis=0)[2]
        logging.info('%s - mean_dice: %.4f, mean_m_ap: %.4f, mean_hd95: %.2f' % (case_name, mean_dice_case, mean_m_ap_case, mean_hd95_case))
    
    metric_array = np.array(metric_list_all)
    
    for i in range(1, args.num_classes):
        class_dice = np.nanmean(metric_array[:, i-1, 0])
        class_m_ap = np.nanmean(metric_array[:, i-1, 1])
        class_hd95 = np.nanmean(metric_array[:, i-1, 2])
        logging.info('Mean class %d - mean_dice: %.4f, mean_m_ap: %.4f, mean_hd95: %.2f' % (i, class_dice, class_m_ap, class_hd95))
        
    mean_dice = np.nanmean(metric_array[:,:,0])
    mean_m_ap = np.nanmean(metric_array[:,:,1])
    mean_hd95 = np.nanmean(metric_array[:,:,2])
    
    logging.info('Testing performance in best val model - mean_dice : %.4f, mean_m_ap : %.4f, mean_hd95 : %.2f' % (mean_dice, mean_m_ap, mean_hd95))
    
    return "Testing Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'COCA': {
            'Dataset': COCA_dataset,
            'volume_path': './data/COCA/test_vol_h5',
            'list_dir': './data/COCA/lists_COCA',
            'num_classes': 4,
            'max_epochs': 300,
            'batch_size': 48,
            'base_lr': 0.00001,
            'img_size': 224,
            'exp_setting': 'sa_window',
            'z_spacing': 3,
        },
    }
    
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.max_epochs = dataset_config[dataset_name]['max_epochs']
    args.batch_size = dataset_config[dataset_name]['batch_size']
    args.base_lr = dataset_config[dataset_name]['base_lr']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.exp_setting = dataset_config[dataset_name]['exp_setting']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']

    args.arch = 'TU'
    snapshot_path = f"./model/{args.arch + '_' + args.vit_name}/{dataset_name + '_' + str(args.img_size)}/{args.exp_setting}/{'epo' + str(args.max_epochs)}"
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    
    net = ViT_seg(config_vit, img_size=args.img_size).cuda()
    best_model_path = glob(os.path.join(snapshot_path, '*_best_model.pth'))[0] # 유일한 best_model 파일 선택
    if not best_model_path:
        raise FileNotFoundError(f"Best model not found at {snapshot_path}")
    net.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from: {best_model_path}")
    
    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = f"./test_log/{args.arch + '_' + args.vit_name}/{dataset_name + '_' + str(args.img_size)}/{args.exp_setting}"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(best_model_path)
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.arch + '_' + args.vit_name, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path)