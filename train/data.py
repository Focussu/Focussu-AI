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
    

class FocusDatasetWithRotation(FocusDataset_multi):
    def __init__(self, base_path, noise=False, both=False, rotation_prob=1.0):
        super().__init__(base_path)
        self.noise = noise
        self.both = both # 기존 데이터셋에 추가해서 증강할건지
        self.rotation_prob = rotation_prob  # 회전 적용 확률

    def get_random_rotation_matrix_xyz(degree_range=15):
        # 각 축마다 -degree_range ~ +degree_range 사이에서 랜덤 각도 선택
        angles = np.radians(np.random.uniform(-degree_range, degree_range, size=3))
        ax, ay, az = angles
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        # # X축 회전 (Pitch)
        # Rx = torch.tensor([
        #     [1, 0, 0],
        #     [0, cx, -sx],
        #     [0, sx,  cx]
        # ])

        # Y축 회전 (Roll) 좌우로 기울이기
        Ry = torch.tensor([
            [cy, 0, sy],
            [0,  1, 0],
            [-sy, 0, cy]
        ], dtype=torch.float32)

        # # Z축 회전 (Yaw)
        # Rz = torch.tensor([
        #     [cz, -sz, 0],
        #     [sz,  cz, 0],
        #     [0,   0,  1]
        # ])

        return Ry

    def __getitem__(self, idx):
        landmark_path = self.meta_df.iloc[idx]['path']
        original_landmarks = torch.from_numpy(np.load(landmark_path)).float()
        label = self.meta_df.iloc[idx]['label']

        apply_rotation = self.noise and (np.random.rand() < self.rotation_prob)

        if apply_rotation:
            rot_matrix = self.get_random_z_rotation_matrix()
            rotated_landmarks = torch.matmul(original_landmarks, rot_matrix)

            if self.both:
                return {
                    "original_landmarks": original_landmarks,
                    "rotated_landmarks": rotated_landmarks,
                    "label": label
                }
            else:
                return {
                    "landmarks": rotated_landmarks,
                    "label": label
                }
        else:
            return {
                "landmarks": original_landmarks,
                "label": label
            }



    
if __name__ == "__main__":
    base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Training'


    dataset = FocusDataset_multi(base_path)
    print(len(dataset))
    print(dataset[0]['label'])