import os
import argparse
import torch
import torch.nn as nn
import timm
import numpy as np
import faiss
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import csv

# --- Import from your existing files ---
try:
    from data_loader import SinglePlantDataLoader
    from quadrat import QuadratTilingDataset_Inference
    import knn as knn_utils 
except ImportError:
    from loading.data_loader import SinglePlantDataLoader
    from loading.quadrat import QuadratTilingDataset_Inference
    import knn as knn_utils

# --- Configuration ---
IMG_SIZE = 224 
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

if "PLANT_HOME" not in os.environ:
    os.environ["PLANT_HOME"] = os.getcwd() 
    print(f"WARNING: PLANT_HOME not set. Defaulting to {os.environ['PLANT_HOME']}")

PLANT_HOME = os.environ["PLANT_HOME"]

class ResNetWrapper(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
    def forward_features(self, x):
        return self.model(x)

def get_model(model_name, device):
    print(f"Loading model: {model_name}...")
    if "resnet" in model_name.lower():
        model = ResNetWrapper(model_name, pretrained=True)
        model.to(device)
        model.eval()
    elif "dino" in model_name.lower():
        checkpoint_path = os.path.join(PLANT_HOME, "dinov2_model/model_best.pth.tar")
        if os.path.exists(checkpoint_path):
            print(f"Loading local DINOv2 checkpoint from {checkpoint_path}")
            model = knn_utils.load_model(device)
        else:
            print("Local checkpoint not found. Downloading from Hub...")
            model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)
            model.to(device)
            model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    return model

def pool_and_aggregate(all_grid_results, min_votes=1):
    """
    Aggregates votes from multiple grid passes.
    
    Args:
        all_grid_results: A list of tuples (tile_preds, tile_paths). 
                          Each tuple is the result of one grid size (e.g., 4x4).
        min_votes: Minimum total votes required to keep a species.
    """
    # Dictionary to hold all votes for each quadrat: { 'quadrat_123': [sp_A, sp_A, sp_B, ...] }
    quadrat_vote_pool = defaultdict(list)
    
    print("\nPooling votes from all grids...")
    
    # 1. Collect votes
    for grid_idx, (preds, paths) in enumerate(all_grid_results):
        print(f"  Processing grid pass {grid_idx + 1}/{len(all_grid_results)} ({len(preds)} tiles)...")
        
        # preds is a list of arrays (neighbors), e.g., [[sp_A, sp_B, sp_A], ...]
        # paths is a list of strings, e.g., ['.../quad_1.jpg', ...]
        
        for tile_neighbors, tile_path in zip(preds, paths):
            # Extract filename as ID (e.g., 'quadrat_123')
            # Using stem to remove extension
            quadrat_id = os.path.splitext(os.path.basename(tile_path))[0]
            
            # Add ALL k-neighbors from this tile to the pool
            # Flattening is important if tile_neighbors is an array
            if isinstance(tile_neighbors, np.ndarray):
                votes = tile_neighbors.flatten().tolist()
            else:
                votes = list(tile_neighbors)
                
            quadrat_vote_pool[quadrat_id].extend(votes)

    # 2. Tally and Threshold
    print(f"  Aggregating votes for {len(quadrat_vote_pool)} unique quadrats...")
    final_predictions = {}
    
    for q_id, all_votes in quadrat_vote_pool.items():
        counter = Counter(all_votes)
        # Keep species that meet the threshold
        species_kept = [sp for sp, count in counter.items() if count >= min_votes]
        
        # Sort for consistency
        final_predictions[q_id] = sorted(species_kept)
        
    return final_predictions

def main():
    parser = argparse.ArgumentParser(description="PlantCLEF 2025 Multi-Grid Pipeline")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "dinov2"], help="Model backbone")
    parser.add_argument("--quadrat_dir", type=str, required=True, help="Folder with UNLABELED quadrats")
    parser.add_argument("--single_plant_dir", type=str, required=True, help="Folder with LABELED single plants")
    
    # NEW: Accept multiple grid sizes
    parser.add_argument("--grids", type=int, nargs='+', default=[4, 5, 6], help="List of grid sizes (e.g., 4 5 6)")
    
    parser.add_argument("--neighbors", type=int, default=5, help="K for KNN")
    parser.add_argument("--votes", type=int, default=3, help="Min TOTAL votes to keep prediction")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Grid Strategy: {args.grids} (pooling votes)")

    # 1. Load Model
    model = get_model(args.backbone, device)

    # 2. Setup Memory Bank (Single Plant Data)
    print("\n--- Setting up Memory Bank (Single Plants) ---")
    single_loader_wrapper = SinglePlantDataLoader(
        data_dir=args.single_plant_dir,
        resize_size=256,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    train_loader, _, _ = single_loader_wrapper.get_dataloaders()
    train_dataset = train_loader.dataset.dataset 
    class_to_speciesid = {i: int(cls_name) for i, cls_name in enumerate(train_dataset.classes)}
    knn_utils.class_to_speciesid = class_to_speciesid 

    # 3. Extract/Load Memory Bank Embeddings (Once)
    emb_dir = os.path.join(PLANT_HOME, "knn", "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    train_emb_file = os.path.join(emb_dir, f"train_embs_{args.backbone}.npz")
    
    if os.path.exists(train_emb_file):
        print(f"Loading cached training embeddings from {train_emb_file}")
        with np.load(train_emb_file) as data:
            train_embs = data['embs']
            train_labels = data['labels']
    else:
        print("Extracting training embeddings...")
        train_embs, train_labels = knn_utils.extract_embeddings(train_loader, train_dataset, model, device)
        np.savez(train_emb_file, embs=train_embs, labels=train_labels)

    # 4. Build FAISS Index (Once)
    print("Normalizing and building Index...")
    faiss.normalize_L2(train_embs)
    index = knn_utils.build_faiss_index(train_embs, device)

    # 5. Multi-Grid Loop
    all_grid_results = [] # Stores (preds, paths) for each grid size

    # Prepare Transform
    from torchvision import transforms
    # Note: We create a fresh transform for inference
    pil_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for grid_n in args.grids:
        print(f"\n--- Processing Grid Size: {grid_n}x{grid_n} ---")
        
        # A. Setup Dataloader for this grid size
        quadrat_dataset = QuadratTilingDataset_Inference(
            data_dir=args.quadrat_dir,
            grid_size=(grid_n, grid_n),
            transform=pil_transform
        )
        quadrat_loader = DataLoader(
            quadrat_dataset,
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )

        # B. Extract/Load Embeddings for this grid
        quad_emb_file = os.path.join(emb_dir, f"quadrat_embs_{args.backbone}_{grid_n}x{grid_n}.npz")
        
        if os.path.exists(quad_emb_file):
            print(f"Loading cached embeddings: {quad_emb_file}")
            with np.load(quad_emb_file) as data:
                quadrat_embs = data['embs']
                quadrat_paths = data['paths']
        else:
            print("Extracting embeddings...")
            quadrat_embs, quadrat_paths = knn_utils.extract_unlabeled_embeddings(quadrat_loader, model, device)
            np.savez(quad_emb_file, embs=quadrat_embs, paths=quadrat_paths)
            
        # C. Run KNN for this grid
        print("Running KNN...")
        faiss.normalize_L2(quadrat_embs)
        tile_preds = knn_utils.knn_predict(quadrat_embs, index=index, faiss_labels=train_labels, k=args.neighbors)
        
        # Store result
        all_grid_results.append((tile_preds, quadrat_paths))

    # 6. Pool and Aggregate
    final_preds = pool_and_aggregate(all_grid_results, min_votes=args.votes)

    # 7. Write Submission
    out_csv = os.path.join(PLANT_HOME, "submissions", f"submission_{args.backbone}_pooled_{'_'.join(map(str, args.grids))}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    knn_utils.write_submission(final_preds, out_csv=out_csv)
    
    print(f"\nDone! Pooled submission saved to: {out_csv}")

if __name__ == "__main__":
    main()
