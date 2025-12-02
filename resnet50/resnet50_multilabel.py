import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import sys
import csv
import zipfile
import json
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Get the ResNet50 Pretrained Model for Transfer Learning
# ============================================================================

def get_resnet50_pretrained(num_classes=10, fine_tune=False):
    """
    Transfer Learning with pretrained models.

    Two strategies:
    1. Feature Extraction (fine_tune=False):
       - Freeze all conv layers (no gradient updates)
       - Only train the new classifier
       - Fast training, good for small datasets
       - Use when: limited data, similar domain

    2. Fine-tuning (fine_tune=True):
       - Initialize with pretrained weights
       - Allow all layers to update
       - Slower training but can achieve higher accuracy
       - Use when: more data, different domain

    Tips:
    - Always replace the final layer to match num_classes
    - When fine-tuning, use lower learning rate (1e-4 or 1e-5)
    - Consider unfreezing only last few layers for better results

    Expected accuracy:
    - Feature extraction: 85-90% in 5 epochs
    - Fine-tuning: 90-95% in 10 epochs
    """
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    for param in model.parameters():  # Freeze all layers
      param.requires_grad=False
    
    if fine_tune == True:   # For fine-tuning, unfreeze the last two layers - 9 Bottlenecks!
        unfrozen_layers = [model.layer3, model.layer4]
        for layer in unfrozen_layers:
            for param in layer.parameters():
                param.requires_grad=True
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)   # Replace final classifier layer
    
    return model  # Return the modified model


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, num_classes, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        multihot_labels = label_to_multihot(labels, num_classes)
        inputs, labels, multihot_labels = inputs.to(device), labels.to(device), multihot_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.sum(criterion(outputs, multihot_labels))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        max_logits, pred_maxidx = outputs.max(1)    # get the max logit and index for each output of the minibatch
        total += labels.size(0)
        correct += pred_maxidx.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, num_classes, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            multihot_labels = label_to_multihot(labels, num_classes)
            inputs, labels, multihot_labels = inputs.to(device), labels.to(device), multihot_labels.to(device)
        
            outputs = model(inputs)
            loss = torch.sum(criterion(outputs, multihot_labels))

            running_loss += loss.item()
            max_logits, pred_maxidx = outputs.max(1)   # get the max logit and index for each output of the minibatch
            total += labels.size(0)
            correct += pred_maxidx.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, num_classes=10, num_epochs=10, lr=0.001, device='cpu'):
    """
    Train and evaluate a model.

    Returns:
        Dictionary with training history
    """
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()   # apply sigmoid for each output logit (not softmax) for multilabel classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    current_path = os.path.abspath('.')
    print('currentpath', current_path)

    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        train_loss, train_acc = train_epoch(model, train_loader, num_classes, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, num_classes, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./resnet50_multilabel_finetune_plantCLEF.pth')
            print(f"\nBest model parameters saved at epoch {epoch+1}!")

        scheduler.step()
    
    return history


def plot_training_history(history, title="Training History"):
    """Plot training and validation loss/accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================================
# Inference / Prediction Functions
# ============================================================================

def multi_label_prediction(model, dataloader, device, threshold=0.5):
    model.eval().to(device)
    all_preds_idx = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc='Predicting'):   # only move imgs to GPU for inference
            imgs = imgs.squeeze().to(device)   # getting rid of the 2nd dim (num_Tiles) since no tiling performed
                                               # for multiple tiles, this is to get rid of the 1st dim since batch_size=1
            logits = model(imgs)
            preds_prob = torch.sigmoid(logits)    # convert to probabilities
            preds_multihot = (preds_prob > threshold).float()  # threshold

            # process results for each batch
            logits = logits.cpu()   # move back to memory
            preds_multihot = preds_multihot.cpu()   # move predictions back to memory
            preds_idx = [np.where(row == 1)[0] for row in preds_multihot]   # get the indices based on multihot
           
            if len(labels) == 1:   # for multiple tiling cases
                quadrat_preds = {}
                quadrat_id = Path(labels[0]).stem
                quadrat_preds[quadrat_id] = preds_idx
                all_preds_idx.append(quadrat_preds)
                
            else:    # for non-tiling
                quadrat_ids = [Path(path).stem for path in labels]   # extract quadrat_id, already in memory   
                batch_preds = dict(zip(quadrat_ids,preds_idx))                           
                all_preds_idx.append(batch_preds)
                
    return all_preds_idx   # return multi-label indices for each image as {quadrat_id: [pred_idx1, pred_idx2, ...]}
                           # results will be a list of dictionaries (each dict represents a minibatch):
                           # for non-tiling, each dictionary contains the quadrat IDs and their predicted indices (idx) of a minibatch
                           # for tiling, each dictionary contains only one quadrat ID and its predicted indices (idx) 


def pred_idx_to_df(all_preds_idx: list, idx_to_cls_mapdict: dict) -> pd.DataFrame:
    '''
    A function to process model prediction/inference results into a dataframe.
    Takes in a list of dictionaries (each dictionary contains quadrat ID(s) and the predicted indices (idx) of a minibatch).
    Returns a DataFrame with three columns: 'quadrat_id', 'species_idx', 'species_ids'.
    '''
    
    result_df = pd.DataFrame(columns=["quadrat_id", "species_idx"])
    ids_col = []
    
    for dict in all_preds_idx:
        if len(dict) == 1:   # for multiple tiling cases
            k, v = next(iter(dict.items()))
            idx_lst = []
            for i in v:   # v is a nested list of list of indices
                if len(i) != 0:
                    for sub_i in i:
                        idx_lst.append(int(sub_i))
            
            df_i = pd.DataFrame({"quadrat_id":k, "species_idx":[idx_lst]})
            result_df = pd.concat([result_df, df_i],ignore_index=True)

        else:   # for non-tiling case
            df_i = pd.DataFrame([[k,v.tolist()] for k, v in dict.items()], columns=["quadrat_id", "species_idx"])
            result_df = pd.concat([result_df, df_i],ignore_index=True)
     
    for quadrat in result_df['species_idx']:
        quadrat_preds = []
        for idx in quadrat:
            cls = idx_to_cls_mapdict[str(idx)]
            if cls not in quadrat_preds:
                quadrat_preds.append(cls)            
        ids_col.append(quadrat_preds)

    result_df['species_ids'] = ids_col
    return result_df


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model):
    """Calculate model size in MB (assuming float32 weights)."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_all_mb    

def label_to_multihot(batch_raw_labels, num_classes=10):
    """Convert minibatch image labels/indices by ImageFolder 
       into multi-hot encoding as Pytorch tensor."""
    batch_multi_hot_labels = []
    
    for labels_per_item in batch_raw_labels:
        item_multi_hot = torch.zeros(num_classes, dtype=torch.float)
        item_multi_hot[labels_per_item] = 1.0
        batch_multi_hot_labels.append(item_multi_hot)

    final_batch_labels = torch.stack(batch_multi_hot_labels)
    return final_batch_labels

    