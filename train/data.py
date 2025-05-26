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

class FocusDataset_V2(Dataset):
    def __init__(self, base_path):
        self.base_path = base_path

        # 저장된 랜드마크 파일 이름 리스트 가져오기
        self.landmark_files = glob.glob(f'{base_path}/01.원천데이터/processed_landmark/*.npy')

        print(f'landmark_files: {len(self.landmark_files)}')
        
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
        
        if len(self.meta_df) == 0:
            raise ValueError("병합 후 데이터가 없습니다. 랜드마크 파일과 메타데이터가 일치하지 않습니다.")

    def __len__(self):
        return len(self.meta_df)
        
    def __getitem__(self, idx):
        landmark_path = self.meta_df.iloc[idx]['path']


        # 저장된 랜드마크와 블렌드쉐이프 로드
        landmarks = torch.from_numpy(np.load(landmark_path)).float()

        
        # 해당 파일명의 라벨 가져오기
        label = self.meta_df.iloc[idx]['category_id']

 
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
        parquet_list = glob.glob(f'{base_path}/02.라벨링데이터/*/*.parquet')
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
    
if __name__ == "__main__":
    base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Training'
    #file_list = ['00_01', '00_02', '00_03', '00_04', '00_05', '10_01',
     #           '10_02','10_03']
    file_list = ['00_01', '10_03', '00_02', '00_03', '10_02']

    dataset = FocusDataset_V2(base_path)
    print(len(dataset))
    print(dataset[0])