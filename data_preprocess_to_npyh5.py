import os
import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm

images_val_dir = '../data/nnUNet_raw/Dataset001_COCA/imagesVal'
labels_val_dir = '../data/nnUNet_raw/Dataset001_COCA/labelsVal'
output_dir = '../data//COCA/test_vol_h5'
os.makedirs(output_dir, exist_ok=True)

val_txt = open('../data/COCA/test_vol.txt', 'w')

image_files = sorted(os.listdir(images_val_dir))

for img_file in tqdm(image_files, desc='Processing', unit='files'):
    # 이미지 파일에서 '_0000.nii.gz'를 제거하여 기본 케이스명을 얻습니다.
    base_name = img_file.replace('_0000.nii.gz', '')
    
    image_path = os.path.join(images_val_dir, img_file)
    label_file = base_name + '.nii.gz'  # 레이블 파일명
    label_path = os.path.join(labels_val_dir, label_file)
    
    if not os.path.exists(label_path):
        print(f"레이블 파일을 찾을 수 없습니다: {label_path}")
        continue
    
    # 케이스 번호 추출 (예: 'COCA_Val_314_0001' → '0001')
    case_number = base_name.split('_')[-1]  # 마지막 '_' 뒤의 숫자 추출
    case_number = case_number.zfill(4)  # 4자리로 패딩
    
    image_3d = nib.load(image_path).get_fdata().astype(np.float32)
    label_3d = nib.load(label_path).get_fdata().astype(np.uint8)
    
    h5_filename = f'case{case_number}.npy.h5'
    with h5py.File(os.path.join(output_dir, h5_filename), 'w') as h5f:
        h5f.create_dataset('image', data=image_3d, compression="gzip", dtype='float32')
        h5f.create_dataset('label', data=label_3d, compression="gzip", dtype='uint8')
    
    val_txt.write(f'case{case_number}\n')

val_txt.close()