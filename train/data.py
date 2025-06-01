from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
import json
import os
from tqdm import tqdm


class FocusDataset(Dataset):
    def __init__(self, base_path, file_list):
        self.base_path = base_path
        self.file_list = file_list
        # 데이터 파일 이름 가져오기
        self.pkl_files = []
        self.meta_files = []
        for file in file_list:
            self.pkl_files.extend(glob.glob(f'{base_path}/01.원천데이터/*{file}*_p/*_result.pkl'))
            self.meta_files.extend(glob.glob(f'{base_path}/02.라벨링데이터/*{file}/*.json'))
        print(f'pkl_files: {len(self.pkl_files)}')
        print(f'meta_files: {len(self.meta_files)}')
       
        # 메타데이터 파일 이름 가져오기
        
        self.meta_dict = {}
        print("메타데이터 로딩 중...")
        for meta_file in tqdm(self.meta_files, desc="메타데이터"):
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
                file_name = meta_data['이미지']['format'].split('.')[0]  # 확장자 제거
                self.meta_dict[file_name] = {
                    'idx': meta_data['이미지']['timeline']['id'],
                    'category_id': meta_data['이미지']['category']['id']
                }


    def __len__(self):
        return len(self.pkl_files)
        
    def __getitem__(self, idx):
        pkl_path = self.pkl_files[idx]

        # pkl 파일명에서 원본 이미지 파일명 추출
        file_name = self.pkl_files[idx].split('/')[-1].replace('_result.pkl', '')

         
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if data.face_landmarks:
            landmarks = torch.tensor([[point.x, point.y, point.z] for point in data.face_landmarks[0]], dtype=torch.float32)
        else:
            landmarks = torch.zeros((478, 3), dtype=torch.float32)
        
        if data.face_blendshapes:   
            blendshapes = torch.tensor([[shape.score for shape in data.face_blendshapes[0]]]        , dtype=torch.float32)
        else:
            blendshapes = torch.zeros((1, 52), dtype=torch.float32)
        
        # 해당 파일명의 라벨 가져오기
        label = self.meta_dict.get(file_name, None)
        
        if label is None:
            raise ValueError(f"라벨을 찾을 수 없습니다: {file_name}")
            
  
        return {
            "file_name": file_name,
            "landmarks": landmarks,
            "blendshapes": blendshapes,
            "label": label
        }

class FocusDataset_multi(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path

        # 저장된 랜드마크 파일 이름 리스트 가져오기
        self.landmark_files = glob.glob(f'{base_path}/01.원천데이터/processed_landmark/*.npy')

        print(f'landmark_files: {len(self.landmark_files)}')
        
        # parquet 메타 데이터 파일 읽기
        meta_df = []
        parquet_list = glob.glob(f'{base_path}/02.라벨링데이터/*_with_emotion/*.parquet')
        CHUNK_SIZE = 1000
        for i in range(0, len(parquet_list), CHUNK_SIZE):
            chunk_files = parquet_list[i:i + CHUNK_SIZE]
            chunk_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
            meta_df.append(chunk_df)
        self.meta_df = pd.concat(meta_df, ignore_index=True)
        self.meta_df['format'] = self.meta_df['format'].str.replace('.jpg', '')
        # 랜드마크 파일 경로와 메타데이터 병합
        landmark_paths_df = pd.DataFrame({
            'path': self.landmark_files,
            'format': [os.path.basename(f).replace('_landmarks.npy', '') for f in self.landmark_files]
        })
        
        # 불일치 확인
        meta_formats = set(self.meta_df['format'])
        landmark_formats = set(landmark_paths_df['format'])
        
        # 메타데이터에만 있는 format
        only_in_meta = meta_formats - landmark_formats
        if only_in_meta:
            print(f"\n경고: 메타데이터에만 존재하는 format이 있습니다:")
            print(f"개수: {len(only_in_meta)}")
            print("예시:", list(only_in_meta)[:5])
        
        # 랜드마크에만 있는 format
        only_in_landmark = landmark_formats - meta_formats
        if only_in_landmark:
            print(f"\n경고: 랜드마크에만 존재하는 format이 있습니다:")
            print(f"개수: {len(only_in_landmark)}")
            print("예시:", list(only_in_landmark)[:5])
        
        # 메타데이터와 랜드마크 경로 병합
        self.meta_df = pd.merge(
            self.meta_df,
            landmark_paths_df,
            on='format',
            how='inner'
        )
        
        print(f'\n병합 전 메타데이터 수: {len(meta_formats)}')
        print(f'병합 전 랜드마크 파일 수: {len(landmark_formats)}')
        print(f'병합 후 데이터 수: {len(self.meta_df)}')
        
        ## 라벨 column 추가
        def create_label(row):
            category_id = row['category_id']
            emotion = row['emotion']
            
            if category_id == 1 and emotion == "흥미로움":
                return 0
            elif category_id == 1 and emotion == "차분함":
                return 1
            elif category_id == 2 and emotion == "차분함":
                return 2
            elif category_id == 2 and emotion == "지루함":
                return 3
            elif category_id == 3:
                return 4
            else:
                # 예외 상황에 대한 처리 (필요시 수정)
                return -1
                
        self.meta_df['label'] = self.meta_df.apply(create_label, axis=1)

        if len(self.meta_df) == 0:
            raise ValueError("병합 후 데이터가 없습니다. 랜드마크 파일과 메타데이터가 일치하지 않습니다.")

    def __len__(self):
        return len(self.meta_df)
        
    def __getitem__(self, idx):
        landmark_path = self.meta_df.iloc[idx]['path']


        # 저장된 랜드마크와 블렌드쉐이프 로드
        landmarks = torch.from_numpy(np.load(landmark_path)).float()

        
        # 해당 파일명의 라벨 가져오기
        label = self.meta_df.iloc[idx]['label']

 
        return {
            "landmarks": landmarks,
            "label": label
        }


class FocusDataset_V3(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path

        # 저장된 랜드마크 파일 이름 리스트 가져오기
        self.landmark_files = glob.glob(f'{base_path}/01.원천데이터/processed_landmark/*.npy')

        print(f'landmark_files: {len(self.landmark_files)}')
        
        # parquet 메타 데이터 파일 읽기
        meta_df = []
        parquet_list = glob.glob(f'{base_path}/02.라벨링데이터/*_with_parts/*.parquet')
        CHUNK_SIZE = 1000
        for i in range(0, len(parquet_list), CHUNK_SIZE):
            chunk_files = parquet_list[i:i + CHUNK_SIZE]
            chunk_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
            meta_df.append(chunk_df)
        self.meta_df = pd.concat(meta_df, ignore_index=True)
        self.meta_df['format'] = self.meta_df['format'].str.replace('.jpg', '')
        # 랜드마크 파일 경로와 메타데이터 병합
        landmark_paths_df = pd.DataFrame({
            'path': self.landmark_files,
            'format': [os.path.basename(f).replace('_landmarks.npy', '') for f in self.landmark_files]
        })
        
        # 불일치 확인
        meta_formats = set(self.meta_df['format'])
        landmark_formats = set(landmark_paths_df['format'])
        self.meta_df['category_id'] = self.meta_df['category_id'].astype(int).map(lambda x: 0 if x == 1 else 1)
        # 메타데이터에만 있는 format
        only_in_meta = meta_formats - landmark_formats
        if only_in_meta:
            print(f"\n경고: 메타데이터에만 존재하는 format이 있습니다:")
            print(f"개수: {len(only_in_meta)}")
            print("예시:", list(only_in_meta)[:5])
        
        # 랜드마크에만 있는 format
        only_in_landmark = landmark_formats - meta_formats
        if only_in_landmark:
            print(f"\n경고: 랜드마크에만 존재하는 format이 있습니다:")
            print(f"개수: {len(only_in_landmark)}")
            print("예시:", list(only_in_landmark)[:5])
        
        # 메타데이터와 랜드마크 경로 병합
        self.meta_df = pd.merge(
            self.meta_df,
            landmark_paths_df,
            on='format',
            how='inner'
        )
        
        print(f'\n병합 전 메타데이터 수: {len(meta_formats)}')
        print(f'병합 전 랜드마크 파일 수: {len(landmark_formats)}')
        print(f'병합 후 데이터 수: {len(self.meta_df)}')
        
        if len(self.meta_df) == 0:
            raise ValueError("병합 후 데이터가 없습니다. 랜드마크 파일과 메타데이터가 일치하지 않습니다.")

    def __len__(self):
        return len(self.meta_df)
        
    def __getitem__(self, idx):
        landmark_path = self.meta_df.iloc[idx]['path']


        # 저장된 랜드마크와 블렌드쉐이프 로드
        landmarks = torch.from_numpy(np.load(landmark_path)).float()

        
        # 해당 파일명의 라벨 가져오기
        label = torch.tensor(self.meta_df.iloc[idx]['category_id'], dtype=torch.float32)

 
        return {
            "landmarks": landmarks,
            "label": label
        }

class blendshape_dataset(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path

        # 저장된 랜드마크 파일 이름 리스트 가져오기
        self.blendshape_files = glob.glob(f'{base_path}/01.원천데이터/processed_blendshape/*.npy')

        print(f'blendshape_files: {len(self.blendshape_files)}')
        
        # parquet 메타 데이터 파일 읽기
        meta_df = []
        parquet_list = glob.glob(f'{base_path}/02.라벨링데이터/*/*.parquet')
        CHUNK_SIZE = 1000
        for i in range(0, len(parquet_list), CHUNK_SIZE):
            chunk_files = parquet_list[i:i + CHUNK_SIZE]
            chunk_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
            meta_df.append(chunk_df)
        self.meta_df = pd.concat(meta_df, ignore_index=True)
        self.meta_df['format'] = self.meta_df['format'].str.replace('.jpg', '')
        # 랜드마크 파일 경로와 메타데이터 병합
        blendshape_paths_df = pd.DataFrame({
            'path': self.blendshape_files,
            'format': [os.path.basename(f).replace('_blendshapes.npy', '') for f in self.blendshape_files]
        })
        
        # 불일치 확인
        meta_formats = set(self.meta_df['format'])
        blendshape_formats = set(blendshape_paths_df['format'])
        self.meta_df['category_id'] = self.meta_df['category_id'].astype(int).map(lambda x: 0 if x == 1 else 1)
        # 메타데이터에만 있는 format
        only_in_meta = meta_formats - blendshape_formats
        if only_in_meta:
            print(f"\n경고: 메타데이터에만 존재하는 format이 있습니다:")
            print(f"개수: {len(only_in_meta)}")
            print("예시:", list(only_in_meta)[:5])
        
        # 랜드마크에만 있는 format
        only_in_blendshape = blendshape_formats - meta_formats
        if only_in_blendshape:
            print(f"\n경고: blendshape에만 존재하는 format이 있습니다:")
            print(f"개수: {len(only_in_blendshape)}")
            print("예시:", list(only_in_blendshape)[:5])
        
        # 메타데이터와 랜드마크 경로 병합
        self.meta_df = pd.merge(
            self.meta_df,
            blendshape_paths_df,
            on='format',
            how='inner'
        )
        
        print(f'\n병합 전 메타데이터 수: {len(meta_formats)}')
        print(f'병합 전 blendshape 파일 수: {len(blendshape_formats)}')
        print(f'병합 후 데이터 수: {len(self.meta_df)}')
        
        if len(self.meta_df) == 0:
            raise ValueError("병합 후 데이터가 없습니다. blendshape 파일과 메타데이터가 일치하지 않습니다.")

    def __len__(self):
        return len(self.meta_df)
        
    def __getitem__(self, idx):
        blendshape_path = self.meta_df.iloc[idx]['path']


        # 저장된 랜드마크와 블렌드쉐이프 로드
        blendshapes = torch.from_numpy(np.load(blendshape_path)).float()

        
        # 해당 파일명의 라벨 가져오기
        label = self.meta_df.iloc[idx]['category_id']
 
        return {
            "blendshapes": blendshapes,
            "label": label
        }
    

class FocusDatasetWithAugmentations(FocusDataset_multi):
    def __init__(self, base_path, rotation=False, noise=False):
        super().__init__(base_path)

        self.use_rotation = rotation
        self.use_noise = noise
        self.rotation_ratio = 0.3
        self.noise_ratio = 0.2
        self.noise_std = 0.01

        self.original_indices = list(range(len(self.meta_df)))

        # 증강 비활성화 시: 원본 데이터만 사용
        if not self.use_rotation and not self.use_noise:
            self.combined_indices = self.original_indices
        else:
            self.rotation_indices = []
            self.noise_indices = []

            if self.use_rotation:
                num_rot = int(len(self.original_indices) * self.rotation_ratio)
                self.rotation_indices = np.random.choice(self.original_indices, size=num_rot, replace=False).tolist()

            if self.use_noise:
                num_noise = int(len(self.original_indices) * self.noise_ratio)
                self.noise_indices = np.random.choice(self.original_indices, size=num_noise, replace=False).tolist()

            # 원본 + 증강된 데이터 추가
            self.combined_indices = self.original_indices + self.rotation_indices + self.noise_indices

    def get_random_y_rotation_matrix(self):
        angle_deg = 5
        theta = np.radians(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return torch.tensor([
            [cos_t, 0, sin_t],
            [0,     1, 0],
            [-sin_t, 0, cos_t]
        ], dtype=torch.float32)

    def add_gaussian_noise(self, landmarks):
        noise = torch.randn_like(landmarks) * self.noise_std
        return landmarks + noise

    def __len__(self):
        return len(self.combined_indices)

    def __getitem__(self, idx):
        original_idx = self.combined_indices[idx]
        landmark_path = self.meta_df.iloc[original_idx]['path']
        landmarks = torch.from_numpy(np.load(landmark_path)).float()
        label = self.meta_df.iloc[original_idx]['label']

        # 회전 적용 여부
        if self.use_rotation and original_idx in self.rotation_indices:
            rot_matrix = self.get_random_y_rotation_matrix()
            landmarks = torch.matmul(landmarks, rot_matrix)

        # 노이즈 적용 여부
        if self.use_noise and original_idx in self.noise_indices:
            landmarks = self.add_gaussian_noise(landmarks)

        return {
            "landmarks": landmarks,
            "label": label
        }




    
if __name__ == "__main__":
    base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Training'


    dataset = FocusDataset_multi(base_path)
    print(len(dataset))
    print(dataset[0]['label'])