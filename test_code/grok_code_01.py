import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 가능한 장치: {device}")
print(f"PyTorch 버전: {torch.__version__}")

# 데이터셋 구조 시뮬레이션
class FloridaWildlifeDatasetSimulator:
    def __init__(self, base_dir='florida_wildlife_dataset'):
        self.base_dir = Path(base_dir)
        self.class_names = [
            'cattle', 'wild_pig', 'white_tailed_deer', 'raccoon', 'bird',
            'opossum', 'rabbit', 'squirrel', 'bobcat', 'chicken', 'horse',
            'crow', 'turkey', 'alligator', 'armadillo', 'otter', 'dog',
            'coyote', 'bear', 'cat', 'florida_panther', 'unknown'
        ]
        self.class_distribution = {
            'cattle': 15000, 'wild_pig': 12000, 'white_tailed_deer': 18000,
            'raccoon': 8000, 'bird': 6000, 'opossum': 5000, 'rabbit': 4000,
            'squirrel': 3500, 'bobcat': 3000, 'chicken': 2500, 'horse': 4500,
            'crow': 2000, 'turkey': 3500, 'alligator': 1500, 'armadillo': 2500,
            'otter': 1000, 'dog': 2000, 'coyote': 1500, 'bear': 800,
            'cat': 1200, 'florida_panther': 2500, 'unknown': 3995
        }
    
    def create_dummy_dataset(self):
        print("데이터셋 구조 생성 중...")
        for split in ['train', 'val', 'test']:
            for class_name in self.class_names:
                (self.base_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'class_names': self.class_names,
            'class_distribution': self.class_distribution,
            'total_images': sum(self.class_distribution.values()),
            'created_date': datetime.now().isoformat()
        }
        
        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"데이터셋 구조 생성 완료: {self.base_dir}")
        return metadata

class WildlifeDataset(Dataset):
    def __init__(self, data_dir, subset='train', img_size=(224, 224), transform=None):
        self.data_dir = Path(data_dir) / subset
        self.transform = transform
        
        if (self.data_dir).exists() and any(self.data_dir.iterdir()):
            self.dataset = torchvision.datasets.ImageFolder(self.data_dir, transform=transform)
        else:
            print(f"경고: {subset} 데이터가 없습니다. 더미 데이터를 생성합니다.")
            self._create_dummy_dataset(img_size)
    
    def _create_dummy_dataset(self, img_size):
        samples_per_class = {'train': 100, 'val': 20, 'test': 30}[subset]
        images = []
        labels = []
        for class_idx, _ in enumerate(self.class_names):
            class_images = torch.randn(samples_per_class, 3, *img_size)
            class_labels = torch.ones(samples_per_class, dtype=torch.long) * class_idx
            images.append(class_images)
            labels.append(class_labels)
        self.images = torch.cat(images)
        self.labels = torch.cat(labels)
    
    def __len__(self):
        if hasattr(self, 'dataset'):
            return len(self.dataset)
        return len(self.images)
    
    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):
            return self.dataset[idx]
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(outputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        pt = torch.exp(-ce_loss)
        alpha_t = torch.ones_like(pt) * alpha
        loss = alpha_t * (1 - pt) ** gamma * ce_loss
        return loss.mean()
    return focal_loss_fixed

def calculate_class_weights(metadata):
    class_counts = list(metadata['class_distribution'].values())
    total_samples = sum(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        weight = total_samples / (len(class_counts) * count)
        if metadata['class_names'][i] == 'florida_panther':
            weight *= 2.0
        class_weights[i] = weight
    return torch.FloatTensor(list(class_weights.values())).to(device)

def train_model(model, train_loader, val_loader, epochs=50, fine_tune_at=None):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
    best_recall = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_recall': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = focal_loss()(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(correct / total)
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = focal_loss()(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = correct / total
        val_recall = recall_score(all_labels, all_preds, average='weighted')
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_recall'].append(val_recall)
        
        print(f"Epoch {epoch+1}, Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {val_acc:.4f}, Val Recall: {val_recall:.4f}")
        
        if val_recall > best_recall:
            best_recall = val_recall
            torch.save(model.state_dict(), 'best_model.pth')
        scheduler.step(val_loss)
    
    return history

# 나머지 코드는 유사하게 구현 가능하며, 시각화 및 평가 부분은 기존 코드와 유사하게 matplotlib, seaborn 사용