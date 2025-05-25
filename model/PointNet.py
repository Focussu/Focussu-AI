import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
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