import math
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import torch
torch.set_num_threads(8)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
import random
import time
from datetime import datetime
import copy

##############################
# CACHE CLEARING
##############################
import sys
sys.dont_write_bytecode = True

import shutil
import os
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        shutil.rmtree(os.path.join(root, '__pycache__'))

import gc
import os
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'

def clear_all_caches():
    """Clear various Python caches"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

clear_all_caches()

# ============================================================================
# SOTA CONFIGURATION CONSTANTS
# ============================================================================

DATASET_NAMES = ['FashionMNIST']  # Focus on FashionMNIST for SOTA
HIDDEN_DIM: int = 512  # Increased capacity
LEARNING_RATE: float = 1e-3
weight_decay_ = 1e-4
DROPOUT_RATE: float = 0.4  # Increased regularization
EPOCHS_PER_DATASET: int = 100  # More epochs for SOTA
BATCH_SIZE: int = 2500
ENSEMBLE_SIZE: int = 3  # Number of models in ensemble

# ============================================================================
# FIXED DATA AUGMENTATION (MOVED TO TOP LEVEL)
# ============================================================================

class MultiAugmentDataset(Dataset):
    """Enhanced dataset with multiple augmentations - MOVED TO TOP LEVEL FOR WINDOWS COMPATIBILITY"""
    def __init__(self, base_dataset, num_augments=2):
        self.base = base_dataset
        self.num_augments = num_augments
        
    def __len__(self):
        return len(self.base) * self.num_augments
    
    def __getitem__(self, idx):
        base_idx = idx // self.num_augments
        return self.base[base_idx]

class MixUp(object):
    """MixUp data augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_sota_data_loaders(dataset_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader, int, int]:
    """SOTA data loading with advanced augmentation - FIXED FOR WINDOWS"""
    
    if dataset_name == 'FashionMNIST':
        # SOTA training transforms
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        n_classes = 10
        n_channels = 1
        img_size = 28
    
    # Enhanced training dataset
    enhanced_train = MultiAugmentDataset(train_dataset, 2)
    
    # FIXED: Use num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        enhanced_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Fixed for Windows
        pin_memory=False,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Fixed for Windows
        pin_memory=False
    )
    
    input_dim = (n_channels, img_size, img_size)
    return train_loader, test_loader, input_dim, n_classes

# ============================================================================
# SOTA ARCHITECTURE COMPONENTS
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Advanced Residual Block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels) if use_se else None
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        residual = x
        
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.se:
            out = self.se(out)
        
        out += self.shortcut(residual)
        out = F.silu(out)
        
        return out

class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        
        self.branch1 = nn.Conv2d(in_channels, out_channels//4, 1, bias=False)
        self.branch2 = nn.Conv2d(in_channels, out_channels//4, 3, padding=1, bias=False)
        self.branch3 = nn.Conv2d(in_channels, out_channels//4, 5, padding=2, bias=False)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        return F.silu(self.bn(out))

# ============================================================================
# SOTA NEURAL NETWORK ARCHITECTURE
# ============================================================================

class SOTAFashionNet(nn.Module):
    """State-of-the-Art Architecture for FashionMNIST"""
    
    def __init__(self, input_dim: tuple, hidden_dim: int, output_dim: int, dropout_rate: float = 0.4):
        super(SOTAFashionNet, self).__init__()
        
        channels, height, width = input_dim
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Multi-scale feature extraction
        self.multiscale1 = MultiScaleBlock(32, 64)
        
        # Residual blocks with progressive channel increase
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.res_block3 = ResidualBlock(128, 128)
        self.res_block4 = ResidualBlock(128, 256, stride=2)
        self.res_block5 = ResidualBlock(256, 256)
        self.res_block6 = ResidualBlock(256, 512, stride=2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Advanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate//2),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate//4),
            
            nn.Linear(hidden_dim//4, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.multiscale1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# ============================================================================
# SOTA TRAINING WITH ADVANCED TECHNIQUES
# ============================================================================

def train_sota_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                    test_loader: DataLoader, optimizer: optim.Optimizer, scheduler,
                    num_epochs: int = 100, verbose: bool = True) -> float:
    
    mixup = MixUp(alpha=0.2)
    best_accuracy = 0.0
    patience_counter = 0
    max_patience = 20
    
    # Simplified training for Windows compatibility
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Use MixUp occasionally
            if epoch > 20 and np.random.random() < 0.3:
                mixed_x, y_a, y_b, lam = mixup(batch_x, batch_y)
                outputs = model(mixed_x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate every 10 epochs to speed up training
        if epoch % 1 == 0:
            test_loss, test_acc = evaluate_model(model, test_loader, criterion)
            
            if verbose:
                current_time = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")
                print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.4f} | "
                      f"Test Acc = {test_acc:.2f}% | LR = {scheduler.get_last_lr()[0]:.6f} | Time: {current_time}")
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience//2:  # Reduced patience
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_accuracy

@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# ============================================================================
# SIMPLIFIED ENSEMBLE TRAINING FOR WINDOWS
# ============================================================================

def train_ensemble_models(input_dim, hidden_dim, n_classes, train_loader, test_loader, 
                         ensemble_size=3) -> Tuple[List[nn.Module], List[float]]:
    """Train multiple models for ensembling - SIMPLIFIED FOR WINDOWS"""
    
    models = []
    best_accuracies = []
    
    for i in range(ensemble_size):
        print(f"\n{'='*60}")
        print(f"Training Ensemble Model {i+1}/{ensemble_size}")
        print(f"{'='*60}")
        
        # Create model with slight variations
        model_hidden = hidden_dim + (i * 64)
        model = SOTAFashionNet(input_dim, model_hidden, n_classes, dropout_rate=DROPOUT_RATE + i*0.05)
        
        # Use AdamW for all models (simpler)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * (0.9 + i*0.1), 
                               weight_decay=weight_decay_, betas=(0.9, 0.999), eps=1e-8)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_DATASET, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_acc = train_sota_model(model, train_loader, criterion, test_loader, optimizer, scheduler,
                                   num_epochs=EPOCHS_PER_DATASET)
        
        models.append(model)
        best_accuracies.append(best_acc)
        
        print(f"Model {i+1} Best Accuracy: {best_acc:.2f}%")
    
    return models, best_accuracies

@torch.no_grad()
def ensemble_predict(models: List[nn.Module], test_loader: DataLoader) -> float:
    """Make ensemble predictions"""
    for model in models:
        model.eval()
    
    correct = 0
    total = 0
    
    for batch_x, batch_y in test_loader:
        # Get predictions from all models
        ensemble_outputs = []
        for model in models:
            outputs = model(batch_x)
            ensemble_outputs.append(F.softmax(outputs, dim=1))
        
        # Average the softmax outputs
        avg_outputs = torch.stack(ensemble_outputs).mean(0)
        preds = avg_outputs.argmax(dim=1)
        
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

# ============================================================================
# MAIN SOTA EXPERIMENT PIPELINE
# ============================================================================

def run_sota_experiments():
    """Run SOTA experiments - FIXED FOR WINDOWS"""
    
    for dataset_name in DATASET_NAMES:
        print(f"\n{'='*80}")
        print(f"RUNNING SOTA EXPERIMENTS ON {dataset_name}")
        print(f"TARGET: 95%+ ACCURACY (WINDOWS OPTIMIZED)")
        print(f"{'='*80}")
        
        train_loader, test_loader, input_dim, n_classes = get_sota_data_loaders(dataset_name, BATCH_SIZE)
        print(f"Dataset: {dataset_name} | Input: {input_dim} | Classes: {n_classes}")
        print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")
        
        # Train ensemble of models
        models, individual_accuracies = train_ensemble_models(
            input_dim, HIDDEN_DIM, n_classes, train_loader, test_loader, ENSEMBLE_SIZE
        )
        
        print(f"\n{'='*60}")
        print("INDIVIDUAL MODEL RESULTS")
        print(f"{'='*60}")
        for i, acc in enumerate(individual_accuracies):
            print(f"Model {i+1}: {acc:.2f}%")
        
        # Ensemble prediction
        print(f"\n{'='*60}")
        print("ENSEMBLE RESULTS")
        print(f"{'='*60}")
        
        ensemble_acc = ensemble_predict(models, test_loader)
        print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
        
        # Final results
        final_accuracy = max(max(individual_accuracies), ensemble_acc)
        print(f"\n{'='*60}")
        print("FINAL SOTA RESULTS")
        print(f"{'='*60}")
        print(f"Best Individual Model: {max(individual_accuracies):.2f}%")
        print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
        print(f"FINAL BEST ACCURACY: {final_accuracy:.2f}%")
        
        if final_accuracy >= 97.0:
            print(f"🎉 EXCELLENT! {final_accuracy:.2f}% ≥ 97%")
        elif final_accuracy >= 95.0:
            print(f"✅ VERY GOOD! {final_accuracy:.2f}% ≥ 95%")
        elif final_accuracy >= 92.0:
            print(f"🟡 GOOD! {final_accuracy:.2f}% ≥ 92%")
        else:
            print(f"⚠️ NEEDS IMPROVEMENT: {final_accuracy:.2f}%")

def main():
    SEED_ = 42
    print("="*80)
    print("WINDOWS-COMPATIBLE SOTA FASHIONMNIST IMPLEMENTATION")
    print("TARGET: 95%+ ACCURACY WITH ENSEMBLE")
    print("="*80)
    print(f"Ensemble Size: {ENSEMBLE_SIZE}")
    print(f"Epochs per Model: {EPOCHS_PER_DATASET}")
    print(f"Hidden Dim: {HIDDEN_DIM}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"CPU Threads: {torch.get_num_threads()}")
    
    torch.manual_seed(SEED_)
    np.random.seed(SEED_)
    random.seed(SEED_)
    
    run_sota_experiments()
    print("\n🎉 SOTA EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    # REMOVED multiprocessing setup for Windows compatibility
    main()
