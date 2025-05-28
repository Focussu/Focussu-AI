from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import  DataLoader
from tqdm import tqdm
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from train.data import blendshape_dataset
from model.PointNet import PointNetClassifier
import numpy as np
import joblib
import os




# 01. 모델 저장하기
# 02. scheduler or weight decay
# 03. Input Normalization 여부 결정하기 (좌표의 범위 확인) -> Normalized좌표라 필요 없음
base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터'
model_save_path = '/home/hyun/focussu-ai/model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
num_epochs = 400
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-4

import matplotlib.pyplot as plt

# 시각화: 상위 N개의 중요 특성
def plot_feature_importances(importances, top_n=10):
    indices = np.argsort(importances)[-top_n:][::-1]
    top_importances = importances[indices]
    feature_names = [f'Feature {i}' for i in indices]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(top_n), top_importances[::-1])
    plt.yticks(range(top_n), feature_names[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Most Important Blendshape Features")
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.show()





def train():
    # 데이터셋 로드
    train_dataset = blendshape_dataset(base_path + '/Training')
    val_dataset = blendshape_dataset(base_path + '/Validation')
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")
    
    # 무작위 샘플링 기반 훈련 데이터 수집
    print("훈련 데이터 무작위 샘플링 중...")
    sample_num = 6000
    indices = random.sample(range(len(train_dataset)), min(sample_num, len(train_dataset)))
    
    X_train = []
    y_train = []
    for idx in tqdm(indices, desc="훈련 데이터 로딩"):
        sample = train_dataset[idx]
        blendshapes = sample['blendshapes'].numpy().flatten()  # (52,)
        label = sample['label']
        X_train.append(blendshapes)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"훈련 데이터 형태: {X_train.shape}")
    print(f"훈련 레이블 형태: {y_train.shape}")
    print(f"훈련 데이터 클래스 분포: {np.bincount(y_train)}")
    
    # 검증 데이터 수집 (전체 로딩)
    print("검증 데이터 수집 중...")
    X_val = []
    y_val = []
    for sample in tqdm(val_dataset, desc="검증 데이터 로딩"):
        blendshapes = sample['blendshapes'].numpy().flatten()
        label = sample['label']
        X_val.append(blendshapes)
        y_val.append(label)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print(f"검증 데이터 형태: {X_val.shape}")
    print(f"검증 레이블 형태: {y_val.shape}")
    print(f"검증 데이터 클래스 분포: {np.bincount(y_val)}")
    
    # Random Forest 모델 훈련
    print("Random Forest 모델 훈련 중...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1  # 모든 CPU 코어 사용
    )
    
    rf_model.fit(X_train, y_train)
    print("모델 훈련 완료!")
    
    # 훈련 데이터에 대한 평가
    print("\n=== 훈련 데이터 평가 ===")
    train_pred = rf_model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()
    print(f"훈련 정확도: {train_accuracy:.4f}")
    print("\n훈련 데이터 Classification Report:")
    print(classification_report(y_train, train_pred))
    
    # 검증 데이터에 대한 평가
    print("\n=== 검증 데이터 평가 ===")
    val_pred = rf_model.predict(X_val)
    val_accuracy = (val_pred == y_val).mean()
    print(f"검증 정확도: {val_accuracy:.4f}")
    print("\n검증 데이터 Classification Report:")
    print(classification_report(y_val, val_pred))
    
    # Confusion Matrix
    print("\n검증 데이터 Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))
    
    # Feature Importance 출력
    feature_importance = rf_model.feature_importances_
    print(f"\n상위 10개 중요한 blendshape 특성:")
    top_features = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_features):
        print(f"{i+1:2d}. Feature {idx:2d}: {feature_importance[idx]:.4f}")
    
    # 모델 저장
    os.makedirs(model_save_path, exist_ok=True)
    
    model_filename = os.path.join(model_save_path, 'random_forest_blendshape.pkl')
    joblib.dump(rf_model, model_filename)
    print(f"\n모델이 저장되었습니다: {model_filename}")
    
    # 성능 요약
    print(f"\n=== 성능 요약 ===")
    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"검증 정확도: {val_accuracy:.4f}")
    print(f"과적합 정도: {train_accuracy - val_accuracy:.4f}")
    # 훈련 후 시각화 호출
    plot_feature_importances(feature_importance, top_n=10)
    return rf_model, train_accuracy, val_accuracy


if __name__ == "__main__":
    train()
