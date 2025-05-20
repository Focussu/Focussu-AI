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
            self.pkl_files.extend(glob.glob(f'{base_path}/01.원천데이터/TS_{file}*_p/*_result.pkl'))
            self.meta_files.extend(glob.glob(f'{base_path}/02.라벨링데이터/TL_{file}/*.json'))
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

            


if __name__ == "__main__":
    base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터/Training'
    #file_list = ['00_01', '00_02', '00_03', '00_04', '00_05', '10_01',
     #           '10_02','10_03']
    file_list = ['00_01', '10_03', '00_02', '00_03', '10_02']

    dataset = FocusDataset(base_path, file_list)
    print(len(dataset))
    print(dataset[0])