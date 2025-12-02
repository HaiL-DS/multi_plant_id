import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import faiss
from pathlib import Path
from collections import Counter, defaultdict
import sys
import os
import csv
from tqdm import tqdm
import json

# Defining project paths
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root) 
    
from resnet50 import resnet50_multilabel
from loading import quadrat_loader

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
training_data_path = os.path.join(project_root, "PlantCLEF2025_data/images_max_side_800")
inference_data_path = os.path.join(project_root, "PlantCLEF2025_data/test_images/images")

# ============================================================================
# 1. ResNet50 Encoder - use the fine tuned ResNet50
# ============================================================================
def build_resnet50_encoder(device):
    # Get the fine tuned ResNet50 multi-label model    
    NUM_CLASS = 7806
    model = resnet50_multilabel.get_resnet50_pretrained(num_classes=NUM_CLASS, fine_tune=False)
    state_dict = torch.load('resnet50_multilabel_finetune_plantCLEF.pth', map_location=torch.device('cpu'), weights_only=True)  
    model.load_state_dict(state_dict)
    
    # Remove classifier → Output the default hidden dim (2048)
    model.fc = nn.Identity()   # output dim = 2048
    model.to(device).eval()
    return model


# ============================================================================
# 2. Feature extraction from single plant training data set
# ============================================================================
# Input batch data shape: (Batch_size x num_Channels x Height x Width)
#                         (     B     x      C       x    H   x   W)                
def extract_features(model, dataloader, device):
    all_feats, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Extracting features ..."):  # imgs: [B, C, H, W], labels: [B, 1]
            imgs = imgs.to(device)
            feats = model(imgs)          # feats: [B, 2048]
            feats = feats.cpu().numpy()

            all_feats.append(feats)
            all_labels.append(labels.numpy())

    feats = np.concatenate(all_feats, axis=0)  # [N, 2048]
    labels = np.concatenate(all_labels, axis=0)  # [N,]
    
    if len(feats.shape[0]) == len(labels) and len(set(labels)) == 7806:
        print(f'Successfully extracted {len(feats.shape[0])} features!\n')
        
    return feats, labels


# ============================================================================
# 3. Build a k-NN index with FAISS
# ============================================================================
def build_faiss_gpu(feats, gpu_id=0):
    dim = feats.shape[1]
    index_flat = faiss.IndexFlatL2(dim)

    # Move to GPU
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, gpu_id, index_flat)

    index.add(feats.astype('float32'))
    print("Successfully built FAISS index with extracted features!\n")
    return index

def save_faiss(index, path="faiss.index"):
    cpu_index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index, path)

def load_faiss(path="faiss.index", gpu_id=0):
    cpu_index = faiss.read_index(path)
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)


# ============================================================================
# 4. Retrieve neighbors (FAISS GPU) 
# ============================================================================
def knn_tile_predict(index, tile_feats, train_labels, k=5):
    distances, indices = index.search(tile_feats.astype(np.float32), k)

    neighbor_labels = train_labels[indices]  # shape: [1,k]
    return neighbor_labels.squeeze()


# ============================================================================
# 5. Aggregate tile predictions → multi-label (multi-hot encoding)
# ============================================================================
def aggregate_tile_predictions(tile_neighbor_labels, threshold=0.25):
    # tile_neighbor_labels = list of arrays from each tile, each shape (k,)
    
    NUM_CLASS = 7806
    vote_counts = np.zeros(NUM_CLASS)

    # Get the quadrat level count for each label/idx
    for tile_votes in tile_neighbor_labels:
        for idx in tile_votes:
            vote_counts[cls] += 1

    # Convert counts into probability-like scores
    scores = vote_counts / vote_counts.max()  # [1, 7806]

    # Threshold (the higher the more stringent)
    threshold = 0.25
    quadrat_preds = (scores >= threshold).astype(int)  # multi-hot: [1, 7806]

    return scores, quadrat_preds


# ============================================================================
# 6. Single plant training dataset & loader
# ============================================================================
def get_train_loader(train_path, batch_size=32):
    # Standard ResNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
            )
        ])

    # Dataset & loaders
    train_data = datasets.ImageFolder(train_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    print(f"Found {len(train_data)} total images belonging to {len(train_data.classes)} classes.")
    print("\n--- Train Dataloader Pipeline Created ---")
    return train_loader  # data shape in each batch: [Batch_size, num_Channels, Height, Width)


# ============================================================================
# 7. Quadrat inference dataset & loader
# ============================================================================
def get_quadrat_loader(quadrat_path, tiles_per_side=3):    
    inference_data, inference_loader = quadrat_loader.main(quadrat_path, 
                                       tiles_per_side,
                                       target_size=224,
                                       batch_size=1, 
                                       num_workers=4)

    print(f"Found {len(inference_data)} total quadrat images.")
    print("\n--- Inference Dataloader Pipeline Created ---")
    return inference_loader  # data shape in each batch: [1, num_Tiles, num_Channels, Height, Width)


# ============================================================================
# 8. Build FAISS index with training data and perform quadrat inference
# ============================================================================
def knn_multi_label_prediction(train_path, quadrat_path, tiles_per_side=3, k=5, threshold=0.25):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset & loaders
    train_loader = get_train_loader(train_path, batch_size=32)
    quadrat_loader = get_quadrat_loader(quadrat_path, tiles_per_side)

    # Build feature extractor
    encoder = build_resnet50_encoder(device=device)

    print("Extracting training features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device=device)
    
    # Build index
    print("Building k-NN index...")
    index = build_faiss_gpu(train_feats, gpu_id=0)
    save_faiss(index, path="faiss.index")

    # Extractor features and make knn predictions from quadrat data
    all_preds_idx = []
    with torch.no_grad():
        for imgs, labels in tqdm(quadrat_loader, desc='Predicting'): 
            imgs = imgs.squeeze().to(device)   # imgs becomes a minibatch of tiles (for one quadrat)
            feats = encoder(imgs).cpu().numpy()

            quadrat_id = Path(labels[0]).stem
            quadrat_preds = {}
            
            # get kNN labels for all tiles of a quadrat
            quadrat_tile_votes = []
            for tile in feats:
                tile_knn_labels = knn_tile_predict(index, tile, train_labels, k)
            quadrat_tile_votes.append(tile_knn_labels)

            scores, quadrat_preds_multihot = aggregate_tile_predictions(quadrat_tile_votes, threshold)
            quadrat_preds_idx = np.where(quadrat_preds_multihot == 1)[0]  # get the indices based on multihot
            
            quadrat_preds[quadrat_id] = quadrat_preds_idx
            all_preds_idx.append(quadrat_preds)

    return all_preds_idx

# ============================================================================
# 9. Implementation with Full pipeline
# ============================================================================
TILES_PER_SIDE = 3
K = 5
THRESHOLD = 0.25

idx_list = knn_multi_label_prediction(training_data_path, 
                                      inference_data_path, 
                                      tiles_per_side=TILES_PER_SIDE, 
                                      k=K, 
                                      threshold=THRESHOLD)

result_df = resnet50_multilabel.pred_idx_to_df(idx_list)

submit_df = result_df[["quadrat_id", "species_ids"]].copy()

submit_df['species_ids'] = submit_df['species_ids'].apply(lambda x: [int(i) for i in x]) 

submit_df.to_csv(f"hl9h_encoder_{TILES_PER_SIDE}x{TILES_PER_SIDE}_thd{THRESHOLD}.csv", sep=',', index=False, quoting=csv.QUOTE_ALL)


