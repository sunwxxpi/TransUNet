import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

images_tr_dir = './data/nnUNet_raw/Dataset001_COCA/imagesTr'
labels_tr_dir = './data/nnUNet_raw/Dataset001_COCA/labelsTr'
output_dir = './data/COCA/train_npz'
os.makedirs(output_dir, exist_ok=True)

train_txt = open('./data/COCA/lists_COCA/train.txt', 'w')

image_files = sorted(os.listdir(images_tr_dir))

for img_file in tqdm(image_files, desc='Processing', unit='files'):
    # 이미지 파일에서 '_0000.nii.gz'를 제거하여 기본 케이스명을 얻습니다.
    base_name = img_file.replace('_0000.nii.gz', '')
    
    image_path = os.path.join(images_tr_dir, img_file)
    label_file = base_name + '.nii.gz'  # 레이블 파일명
    label_path = os.path.join(labels_tr_dir, label_file)
    
    if not os.path.exists(label_path):
        print(f"레이블 파일을 찾을 수 없습니다: {label_path}")
        continue
    
    # 케이스 번호 추출 (예: 'COCA_Tr_0_0001' → '0001')
    case_number = base_name.split('_')[-1]  # 마지막 '_' 뒤의 숫자 추출
    case_number = case_number.zfill(4)  # 4자리로 패딩
    
    image_3d = nib.load(image_path).get_fdata().astype(np.float32)
    label_3d = nib.load(label_path).get_fdata().astype(np.float32)
    num_slices = image_3d.shape[2]
    
    for slice_idx in range(num_slices):
        image_slice = image_3d[:, :, slice_idx]
        label_slice = label_3d[:, :, slice_idx]
        npz_filename = f'case{case_number}_slice{slice_idx:03d}.npz'
        np.savez(os.path.join(output_dir, npz_filename), image=image_slice, label=label_slice)
        
        train_txt.write(f'case{case_number}_slice{slice_idx:03d}\n')

train_txt.close()