import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

label_map = {
    0: "집중",
    1: "비집중",
    2: "졸음"
}

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(PointNetClassifier, self).__init__()

        self.fc1 = nn.Linear(3, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.cls_fc1 = nn.Linear(256, 128)
        self.cls_bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.cls_fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        B, N, D = x.size()  # (B, 478, 3)

        x = self.fc1(x)  # (B, N, 64)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)

        x = self.fc2(x)  # (B, N, 128)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)

        x = self.fc3(x)  # (B, N, 256)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)

        x = torch.max(x, dim=1)[0]  # (B, 256)

        x = self.cls_fc1(x)         # (B, 128)
        x = self.cls_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.cls_fc2(x)         # (B, num_classes)
        return x


def predict(model, data, device):
    model.eval()
    
    # 배치 차원이 없으면 추가 (478, 3) -> (1, 478, 3)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    
    # UserWarning 해결: torch.tensor() 대신 detach().clone() 사용
    if data.device != device:
        landmarks = data.detach().clone().to(device)
    else:
        landmarks = data.detach().clone()

    with torch.no_grad():
        logits = model(landmarks)
        probs = torch.sigmoid(logits)
        confidence = (1-probs[0].item())
        return round(confidence, 4)

def load_pointnet(model_path, device):
    model = PointNetClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


