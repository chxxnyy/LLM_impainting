import numpy as np
import h5py

def read_dataset(file_path, data_len):
    img_list = []
    gt_list = []
    
    with h5py.File(file_path, 'r') as file:
        # 각 데이터셋 로드
        data_img = file['ih']
        data_gt = file['b_']
        
        # 길이만큼 데이터 불러오기
        for i in range(data_len):
            img_list.append(np.array(data_img[i]))
            gt_list.append(np.array(data_gt[i]))
    
    return img_list, gt_list