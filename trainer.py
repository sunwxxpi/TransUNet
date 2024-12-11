import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
from utils import PolyLRScheduler, DiceLoss
from datasets.dataset import shuffle_within_batch, COCA_dataset, RandomGenerator, Resize, ToTensor

def save_attention_maps(attention_maps, image_ids, epoch, save_dir='attention_maps'):
    """
    배치 내의 모든 슬라이스에 대한 어텐션 맵을 저장하는 함수.

    Args:
        attention_maps (dict): 'self', 'prev', 'next' 키를 가지는 딕셔너리이며, 각 값은 어텐션 맵 텐서입니다.
        image_ids (list): 이미지 식별자의 리스트 (배치 크기만큼의 길이).
        epoch (int): 현재 에폭 번호.
        save_dir (str): 어텐션 맵을 저장할 디렉토리 경로.
    """
    # 에폭별 디렉토리 생성
    save_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 배치 크기 및 기타 정보 추출
    batch_size = attention_maps['self'].size(0)  # 배치 크기
    num_heads = attention_maps['self'].size(1)   # 헤드 수
    N = attention_maps['self'].size(2)           # 토큰 개수 (예: H*W)
    h = w = int(np.sqrt(N))                      # 패치맵의 높이와 너비

    # 배치 내의 모든 슬라이스에 대해 반복
    for slice_index in range(batch_size):
        image_id = image_ids[slice_index]  # 현재 슬라이스의 이미지 식별자
        # 각 슬라이스에 대한 어텐션 맵 저장
        attention_types = ['prev', 'self', 'next']
        attention_maps_list = []
        for attn_type in attention_types:
            attn_map = attention_maps[attn_type][slice_index]  # (num_heads, N, N)
            # 헤드별로 어텐션 맵을 평균화
            attn_map_avg = attn_map.mean(dim=0)  # (N, N)
            # 열 방향으로 평균 계산 (쿼리 위치에 대한 전체 키 위치의 평균)
            attn_map_mean = attn_map_avg.mean(dim=0).cpu().numpy()  # (N,)
            # 어텐션 맵을 2D 이미지로 변환
            attn_map_img = attn_map_mean.reshape(h, w)
            # 어텐션 맵 정규화
            attn_map_norm = (attn_map_img - attn_map_img.min()) / (attn_map_img.max() - attn_map_img.min() + 1e-8)
            attention_maps_list.append(attn_map_norm)
        
        # 세 개의 어텐션 맵을 하나의 이미지로 합치기
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Prev Slice Attention', 'Self-Attention', 'Next Slice Attention']
        vmin, vmax = 0.0, 1.0
        for i, attn_map in enumerate(attention_maps_list):
            im = axes[i].imshow(attn_map, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
            axes[i].set_title(titles[i])
            axes[i].axis('off')
            # 컬러바 추가
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        plt.suptitle(f'Epoch {epoch}, Image ID: {image_id}')
        # 이미지 식별자를 파일명에 포함
        filename = f'{image_id}.png'
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

def trainer_coca(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", 
                        level=logging.INFO, 
                        format='[%(asctime)s.%(msecs)03d] %(message)s', 
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    
    train_transform = RandomGenerator(output_size=[args.img_size, args.img_size])
    db_train = COCA_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=train_transform
    )
    
    val_transform = T.Compose([
        Resize(output_size=[args.img_size, args.img_size]),
        ToTensor()
    ])
    db_val = COCA_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        transform=val_transform
    )

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False, 
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, 
                             collate_fn=shuffle_within_batch)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.train()
    
    dice_loss_class = DiceLoss(num_classes)
    ce_loss_class = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    # optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    
    max_iterations = args.max_epochs * len(trainloader)
    scheduler = PolyLRScheduler(optimizer, initial_lr=base_lr, max_steps=max_iterations)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iter_num = 0
    max_epoch = args.max_epochs
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch_num in tqdm(range(1, max_epoch + 1), ncols=70):
        train_dice_loss = 0.0
        train_ce_loss = 0.0
        train_loss = 0.0
        
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch['image'].cuda()
            prev_image_batch = sampled_batch['prev_image'].cuda()
            next_image_batch = sampled_batch['next_image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            case_name = sampled_batch['case_name']
            
            slice_inputs = (prev_image_batch, image_batch, next_image_batch)
            outputs, attention_maps, attn_weights = model(slice_inputs, return_attn=True)

            dice_loss = dice_loss_class(outputs, label_batch, softmax=True)
            ce_loss = ce_loss_class(outputs, label_batch)
            loss = (0.5 * dice_loss) + (0.5 * ce_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            
            iter_num += 1
            
            train_dice_loss += dice_loss.item()
            train_ce_loss += ce_loss.item()
            train_loss += loss.item()

            logging.info('epoch %d, iteration %d - dice_loss: %f, ce_loss: %f, loss_total: %f' % (epoch_num, iter_num, dice_loss.item(), ce_loss.item(), loss.item()))
        
        # TensorBoard에 기록
        center_image = image_batch[1, 0:1, :, :]
        center_image = (center_image - center_image.min()) / (center_image.max() - center_image.min())
        pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        labels = label_batch[1, ...].unsqueeze(0) * 50
        
        writer.add_image('train/Center_Image', center_image, iter_num)
        writer.add_image('train/Prediction', pred[1, ...] * 50, iter_num)
        writer.add_image('train/GroundTruth', labels, iter_num)

        train_dice_loss /= len(trainloader)
        train_ce_loss /= len(trainloader)
        train_loss /= len(trainloader)
        
        writer.add_scalar('train/lr', current_lr, epoch_num)
        writer.add_scalar('train/dice_loss', train_dice_loss, epoch_num)
        writer.add_scalar('train/ce_loss', train_ce_loss, epoch_num)
        writer.add_scalar('train/train_loss', train_loss, epoch_num)
        logging.info('Train - epoch %d - train_dice_loss: %f, train_ce_loss: %f, train_loss: %f' % (epoch_num, train_dice_loss, train_ce_loss, train_loss))
        
        # Attention Maps 저장
        # save_attention_maps(attention_maps, case_name, epoch_num)

        # Validation step after each epoch
        val_dice_loss = 0.0
        val_ce_loss = 0.0
        val_loss = 0.0
        
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                image_batch = sampled_batch['image'].cuda()
                prev_image_batch = sampled_batch['prev_image'].cuda()
                next_image_batch = sampled_batch['next_image'].cuda()
                label_batch = sampled_batch['label'].cuda()
                
                slice_inputs = (prev_image_batch, image_batch, next_image_batch)
                outputs = model(slice_inputs)
                
                dice_loss = dice_loss_class(outputs, label_batch, softmax=True)
                ce_loss = ce_loss_class(outputs, label_batch)
                loss = (0.5 * dice_loss) + (0.5 * ce_loss)
                
                val_dice_loss += dice_loss.item()
                val_ce_loss += ce_loss.item()
                val_loss += loss.item()

        # Epoch별 평균 validation 손실 계산 및 기록
        val_dice_loss /= len(valloader)
        val_ce_loss /= len(valloader)
        val_loss /= len(valloader)

        writer.add_scalar('val/dice_loss', val_dice_loss, epoch_num)
        writer.add_scalar('val/ce_loss', val_ce_loss, epoch_num)
        writer.add_scalar('val/val_loss', val_loss, epoch_num)
        logging.info('Validation - epoch %d - val_dice_loss: %f, val_ce_loss: %f, val_loss: %f' % (epoch_num, val_dice_loss, val_ce_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_model_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_{best_val_loss:.4f}_best_model.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), best_model_path)
            else:
                torch.save(model.state_dict(), best_model_path)
                
            logging.info(f"Best model saved to {best_model_path} with val_loss: {best_val_loss:.4f}")

        if epoch_num == max_epoch:
            save_model_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_{val_loss:.4f}.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_model_path)
            else:
                torch.save(model.state_dict(), save_model_path)
                
            logging.info(f"Final epoch model saved to {save_model_path} with val_loss: {val_loss:.4f}")

    writer.close()
    return "Training Finished!"