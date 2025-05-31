from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from tqdm import tqdm
import wandb

from train.data import FocusDataset_multi
from model.PointNet import PointNetClassifier

# 01. 모델 저장하기
# 02. scheduler or weight decay
# 03. Input Normalization 여부 결정하기 (좌표의 범위 확인) -> Normalized좌표라 필요 없음
base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터'
model_save_path = '/home/hyun/focussu-ai/model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetClassifier(num_classes=5).to(device)
num_epochs = 400
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-4
def train():
    # wandb 초기화
    wandb.login()
    run = wandb.init(
        project="focus-classification",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "model": "PointNetClassifier",
            "weight_decay": weight_decay
        }
    )
    
    train_dataset = FocusDataset_multi(base_path + '/Training')
    val_dataset = FocusDataset_multi(base_path + '/Validation')

    train_loader = DataLoader(train_dataset,num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    
    # 옵션 1: ReduceLROnPlateau - validation loss가 개선되지 않을 때만 lr 감소
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 옵션 2: 더 안정적인 CyclicLR (lr 변화폭 감소, 상승 구간 단축)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=0.0005, max_lr=0.003, step_size_up=20, step_size_down=80, mode='triangular2'
    # )
    
    # 옵션 3: OneCycleLR - 한 번의 cycle 후 낮은 lr로 유지
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=num_epochs,
    #     pct_start=0.3, div_factor=10, final_div_factor=100
    # )
    
    # 옵션 4: CosineAnnealingLR - 부드러운 감소
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_accuracy = 0.0
    best_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f'Batches (Epoch {epoch+1})', leave=False)):
            landmarks = batch['landmarks'].to(device)
            label = batch['label'].long().to(device)  # long tensor로 변경하고 unsqueeze 제거
            optimizer.zero_grad()
            output = model(landmarks)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            # OneCycleLR을 사용하는 경우에만 매 step마다 scheduler.step() 호출
            # if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            #     scheduler.step()


        avg_train_loss = train_loss / len(train_loader)
        wandb.log({
            "train_loss": avg_train_loss,
        })
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    landmarks = batch['landmarks'].to(device)
                    label = batch['label'].long().to(device)  # long tensor로 변경하고 unsqueeze 제거
                    output = model(landmarks)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                accuracy = 100 * correct / total
                print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}%, Val Loss: {avg_val_loss:.4f}')
                
                # ReduceLROnPlateau의 경우 validation loss를 기준으로 step
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                # 다른 scheduler들의 경우 epoch 기준으로 step
                elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                
                if(accuracy > best_accuracy):
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), model_save_path + '/best_multi_model.pth')
                    
                # validation loss 기준으로도 모델 저장
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), model_save_path + '/best_multi_model_loss.pth')
                    
                # wandb에 로깅
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/accuracy": accuracy,
                    "lr": current_lr
                })
    
    wandb.finish()

if __name__ == '__main__':
    train()



