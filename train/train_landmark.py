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

from train.data import FocusDataset_V2
from model.PointNet import PointNetClassifier

# 01. 모델 저장하기
# 02. scheduler or weight decay
# 03. Input Normalization 여부 결정하기 (좌표의 범위 확인) -> Normalized좌표라 필요 없음
base_path = '/shared_data/focussu/109.학습태도_및_성향_관찰_데이터/3.개방데이터/1.데이터'
model_save_path = '/home/hyun/focussu-ai/model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNetClassifier(num_classes=1).to(device)
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
    
    train_dataset = FocusDataset_V2(base_path + '/Training')
    val_dataset = FocusDataset_V2(base_path + '/Validation')

    train_loader = DataLoader(train_dataset,num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=50, step_size_down=100, mode='triangular')
    best_accuracy = 0.0

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f'Batches (Epoch {epoch+1})', leave=False)):
            landmarks = batch['landmarks'].to(device)
            label = (batch['label'] - 1).float().to(device)
            optimizer.zero_grad()
            output = model(landmarks)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()


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
                    label = (batch['label'] - 1).to(device)  # Train과 동일한 방식
                    output = model(landmarks)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    #### 
                    predicted = torch.round(torch.sigmoid(output))
                    correct += (predicted == label).sum().item()
                    total += label.size(0)
                    #_, predicted = torch.max(output.data, 1)
                    #total += label.size(0)
                    #correct += (predicted == label).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                accuracy = 100 * correct / total
                print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy}%')
                if(accuracy > best_accuracy):
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), model_save_path + '/best_model.pth')
                # wandb에 로깅
                wandb.log({
                    "val/loss": avg_val_loss,
                    "val/accuracy": accuracy,
                    "lr": scheduler.get_last_lr()[0]
                })
    
    wandb.finish()

if __name__ == '__main__':
    train()



