import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
from collections import Counter
from torch.utils.data import Subset
from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
import random
from PIL import Image

# Defining project paths
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root) 
    
training_eda_path = os.path.join(project_root, "eda_img")
training_data_path = os.path.join(project_root, "PlantCLEF2025_data/images_max_side_800")
inference_data_path = os.path.join(project_root, "PlantCLEF2025_data/test_images/images")

from loading import data_loader   # get our own dataloader to preprocess input images for ResNet50
from resnet50 import resnet50_multilabel   # get pretrained ResNet50 and training functions

# Create an instance of the SinglePlantDataLoader
RESIZE_SIZE = 225
IMG_SIZE = 224 
BATCH_SIZE = 32
NUM_WORKERS = 4   # Use all available CPU cores for loading os.cpu_count()
# NUM_CLASSES = 7806   # As per the challenge overview, confirm this by self.classes

data_splitter = data_loader.SinglePlantDataLoader(
                                    data_dir=training_data_path,
                                    resize_size=RESIZE_SIZE,
                                    img_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    train_split=0.8, val_split=0.2, test_split=0.0
                                    )

# Get the dataloaders
train_loader, val_loader, test_loader = data_splitter.get_dataloaders()
print("\n--- Dataloader Pipelines Created ---")

# Create an index-to-name mapping for the full training data
idx_to_class_mapdict = {idx: class_name for class_name, idx in data_splitter.class_to_idx.items()}

# Initialize configuations for training a multi-label head
NUM_CLASS = 7806
NUM_EPOCH = 30
LR = 0.001
DEVICE = resnet50_multilabel.device
print(f'Using device: {DEVICE}')

# Train a Multilabel Classification Head on ResNet50 with PlantCLEF Training Data
print("\n" + "="*60)
print("Transfer Learning - Train a Multilabel Classification Head on ResNet50 with PlantCLEF Training Data")
print("="*60)

print("\nTraining ResNet50 ...")

pretrained_resnet = resnet50_multilabel.get_resnet50_pretrained(num_classes=NUM_CLASS)
num_params, trainable_params = resnet50_multilabel.count_parameters(pretrained_resnet)
model_size = resnet50_multilabel.get_model_size_mb(pretrained_resnet)
print(f"Total parameters: {num_params:,d}")
print(f"Trainable parameters: {trainable_params:,d}")
print(f"Model size: {model_size:.2f} MB")

train_history = resnet50_multilabel.train_model(pretrained_resnet, 
                               train_loader, 
                               val_loader,
                               num_classes=NUM_CLASS,
                               num_epochs=NUM_EPOCH, 
                               lr=LR,
                               device=DEVICE)

results_df = pd.DataFrame(train_history)
results_df.to_csv(f'./train_history.csv', index=False)
print(f"\nTraining {NUM_EPOCH} epochs finished! Training history saved to 'train_history.csv'")















