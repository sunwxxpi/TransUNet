import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if np.random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif np.random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class COCA_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, train_ratio=0.8):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir

        # train.txt 파일을 읽어 전체 리스트를 가져옵니다.
        full_sample_list = open(os.path.join(list_dir, "train.txt")).readlines()

        if split in ["train", "val"]:
            train_samples, val_samples = train_test_split(full_sample_list, train_size=train_ratio, random_state=42)
            self.sample_list = train_samples if split == "train" else val_samples
        else:
            self.sample_list = open(os.path.join(list_dir, "test_vol.txt")).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')
        
        if self.split == "train" or self.split == "val":
            # train과 val의 경우 .npz 파일을 사용합니다.
            data_path = os.path.join(self.data_dir, sample_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            # test의 경우 .npy.h5 파일을 사용합니다.
            filepath = os.path.join(self.data_dir, "{}.npy.h5".format(sample_name))
            data = h5py.File(filepath, 'r')
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = sample_name
        return sample