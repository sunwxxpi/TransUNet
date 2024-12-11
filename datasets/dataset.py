import os
import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as T
from scipy import ndimage
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

def random_rot_flip(image, label):
    """Randomly rotate and flip the image and label."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    if label is not None:
        label = np.rot90(label, k)
    
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.flip(label, axis=axis).copy()
    
    return image, label

def random_rotate(image, label):
    """Randomly rotate the image and label by a small angle."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    
    return image, label

def ct_normalization(image, lower=1016, upper=1807, mean=1223.2043595897762, std=133.03651991499345):
    """Normalize the CT image using fixed intensity range and standardization."""
    np.clip(image, lower, upper, out=image)
    image = (image - mean) / max(std, 1e-8)
    
    return image

def fixed_min_max_normalization(image, min_val=0, max_val=2500):
    """Normalize the image based on fixed min and max values of 0 and 2500."""
    normalized_img = (image - min_val) / (max_val - min_val)
    
    return np.clip(normalized_img, 0, 1)

def shuffle_within_batch(batch):
    random.shuffle(batch)
    
    return default_collate(batch)

class RandomAugmentation:
    """Apply random rotations and flips to the image, prev_image, next_image, and label."""
    def __call__(self, sample):
        image, prev_image, next_image, label = sample['image'], sample['prev_image'], sample['next_image'], sample['label']
        
        # Apply random rotation and flip consistently across image, prev_image, next_image, and label
        if np.random.rand() > 0.5:
            image, label = random_rot_flip(image, label)
            prev_image, _ = random_rot_flip(prev_image, None)
            next_image, _ = random_rot_flip(next_image, None)
            
        if np.random.rand() > 0.5:
            image, label = random_rotate(image, label)
            prev_image, _ = random_rotate(prev_image, None)
            next_image, _ = random_rotate(next_image, None)
            
        sample['image'], sample['prev_image'], sample['next_image'], sample['label'] = image, prev_image, next_image, label
        return sample

class Resize:
    """Resize the image, prev_image, next_image, and label to the desired output size."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, prev_image, next_image, label = sample['image'], sample['prev_image'], sample['next_image'], sample['label']
        x, y = image.shape
        
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            prev_image = zoom(prev_image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            next_image = zoom(next_image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        sample['image'], sample['prev_image'], sample['next_image'], sample['label'] = image, prev_image, next_image, label
        return sample

class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    def __call__(self, sample):
        image, prev_image, next_image, label = sample['image'], sample['prev_image'], sample['next_image'], sample['label']
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        prev_image = torch.from_numpy(prev_image.astype(np.float32)).unsqueeze(0)
        next_image = torch.from_numpy(next_image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        
        sample['image'], sample['prev_image'], sample['next_image'], sample['label'] = image, prev_image, next_image, label
        return sample

class RandomGenerator:
    """Compose random augmentations and preprocessing for training."""
    def __init__(self, output_size):
        self.transform = T.Compose([
            RandomAugmentation(),
            Resize(output_size),
            ToTensor()
        ])

    def __call__(self, sample):
        return self.transform(sample)

class COCA_dataset(Dataset):
    """Custom dataset for COCA data with previous and next images."""
    def __init__(self, base_dir, list_dir, split, transform=None, train_ratio=0.8):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir

        # 학습 및 검증 데이터 로드 (2D 슬라이스)
        if split in ["train", "val"]:
            with open(os.path.join(list_dir, "train.txt"), 'r') as f:
                full_sample_list = f.readlines()
            train_samples, val_samples = train_test_split(full_sample_list, train_size=train_ratio, shuffle=False, random_state=42)
            self.sample_list = train_samples if split == "train" else val_samples
        else:
            # 테스트 데이터 로드 (3D 볼륨)
            with open(os.path.join(list_dir, "test_vol.txt"), 'r') as f:
                self.sample_list = sorted(f.readlines())

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')
        
        if self.split in ["train", "val"]:
            # 2D 슬라이스 단위로 데이터 로드
            volume_name = sample_name.split('_slice')[0]
            volume_slices = [s.strip('\n') for s in self.sample_list if s.startswith(volume_name)]
            current_index = volume_slices.index(sample_name)
            
            # 볼륨 내에서 인접한 슬라이스를 가져옴
            prev_sample_name = volume_slices[max(current_index - 1, 0)]
            next_sample_name = volume_slices[min(current_index + 1, len(volume_slices) - 1)]

            # 현재, 이전, 다음 슬라이스 로드
            data_path = os.path.join(self.data_dir, sample_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            
            prev_image = np.load(os.path.join(self.data_dir, prev_sample_name + '.npz'))['image']
            next_image = np.load(os.path.join(self.data_dir, next_sample_name + '.npz'))['image']
        else:
            # 테스트 모드에서 3D 볼륨 단위로 데이터 로드
            filepath = os.path.join(self.data_dir, f"{sample_name}.npy.h5")
            with h5py.File(filepath, 'r') as data:
                image, label = data['image'][:], data['label'][:]

            # prev_image와 next_image를 인접 슬라이스 참조 방식으로 생성
            prev_image = np.copy(image)
            next_image = np.copy(image)

            # 맨 윗 슬라이스의 prev_image를 첫 슬라이스로 고정
            prev_image[1:] = image[:-1]  # 첫 슬라이스를 제외한 나머지 슬라이스가 이전 슬라이스를 참조
            prev_image[0] = image[0]     # 맨 윗 슬라이스는 자기 자신을 참조

            # 맨 아랫 슬라이스의 next_image를 마지막 슬라이스로 고정
            next_image[:-1] = image[1:]  # 마지막 슬라이스를 제외한 나머지 슬라이스가 다음 슬라이스를 참조
            next_image[-1] = image[-1]   # 맨 아랫 슬라이스는 자기 자신을 참조
            
        image = ct_normalization(image)
        prev_image = ct_normalization(prev_image)
        next_image = ct_normalization(next_image)
        
        # `train` 및 `val` 모드에서는 2D 슬라이스를, `test` 모드에서는 3D 볼륨 전체를 반환
        sample = {
            'image': image, 
            'prev_image': prev_image, 
            'next_image': next_image,
            'label': label,
            'case_name': sample_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample