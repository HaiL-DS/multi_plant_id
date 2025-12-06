import os
import sys

# --- 1. CRITICAL SETUP ---
PLANT_HOME = "/scratch/ezq9qu"
os.environ["PLANT_HOME"] = PLANT_HOME

# --- 2. FIX PYTHON PATH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- 3. IMPORTS ---
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import faiss
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from torchvision import transforms
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from loading.data_loader import SinglePlantDataLoader
    from loading.quadrat import QuadratTilingDataset_Inference
    from knn import knn as knn_utils 
except ImportError:
    from data_loader import SinglePlantDataLoader
    from quadrat import QuadratTilingDataset_Inference
    import knn as knn_utils

# --- CONFIGURATION ---
BASE_BATCH_SIZE = 32
NUM_WORKERS = 4

# --- MODEL WRAPPERS ---
class ResNetWrapper(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    def forward_features(self, x):
        return self.model(x) 

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
    def forward(self, x):
        if hasattr(self.module, 'forward_features'):
            feats = self.module.forward_features(x)
        else:
            feats = self.module(x)
        if feats.ndim == 3: return feats[:, 0, :]
        return feats

def get_model(model_name, device, num_classes=None, use_lora=False, lora_checkpoint=None):
    print(f"Loading model: {model_name}...")
    
    if "resnet" in model_name.lower():
        if num_classes:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        else:
            model = ResNetWrapper(model_name, pretrained=True)
            
    elif "dino" in model_name.lower():
        if num_classes:
            model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=num_classes)
        else:
            model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)

    model.to(device)

    # 2. Apply New LoRA (Training)
    if use_lora and num_classes and not lora_checkpoint:
        if not PEFT_AVAILABLE: raise ImportError("pip install peft")
        print("Applying NEW LoRA adapters for training...")
        target_modules = ["qkv"] if "dino" in model_name else ["conv1", "conv2"] 
        peft_config = LoraConfig(inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 3. Load Existing LoRA (Inference)
    if lora_checkpoint:
        if not PEFT_AVAILABLE: raise ImportError("pip install peft")
        
        # --- ROBUST PATH CHECKING ---
        config_path = os.path.join(lora_checkpoint, "adapter_config.json")
        if not os.path.exists(config_path):
            print(f"\nERROR: 'adapter_config.json' not found at {config_path}")
            
            # Check parent folder for clues (e.g. lowercase issues)
            parent = os.path.dirname(lora_checkpoint.rstrip('/'))
            try:
                available = os.listdir(parent)
                print(f"Contents of {parent}: {available}")
                
                # Check for lowercase match
                target_name = os.path.basename(lora_checkpoint.rstrip('/')).lower()
                for name in available:
                    if name.lower() == target_name:
                        print(f"Did you mean: {os.path.join(parent, name)}?")
            except Exception:
                print(f"Cannot list contents of {parent}")
                
            raise FileNotFoundError(f"LoRA config not found at {lora_checkpoint}")
            
        print(f"Loading LoRA adapters from: {lora_checkpoint}")
        if "dino" in model_name and not num_classes:
             print("Warning: Loading LoRA on DINOv2 without a classification head.")

        model = PeftModel.from_pretrained(model, lora_checkpoint)
        try:
            model = model.merge_and_unload()
            print("LoRA weights merged into backbone.")
        except Exception as e:
            print(f"Note: Kept LoRA adapters separate ({e})")

    return model

# --- PARALLEL EXTRACTION FUNCTIONS ---
def extract_embeddings_parallel(dataloader, model, device, class_to_speciesid):
    all_embs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, idx in tqdm(dataloader, desc="Extracting Train Embeddings"):
            imgs = imgs.to(device)
            cls_embs = model(imgs) 
            true_ids = [class_to_speciesid[int(c)] for c in idx]
            all_embs.append(cls_embs.cpu().numpy())
            all_labels.append(true_ids)
    return np.concatenate(all_embs), np.concatenate(all_labels)

def extract_unlabeled_embeddings_parallel(dataloader, model, device):
    all_embs, all_paths = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Quadrat Embeddings"):
            imgs = batch[0].to(device)
            paths = batch[1] 
            cls_embs = model(imgs)
            all_embs.append(cls_embs.cpu().numpy())
            all_paths.extend(paths)
    return np.concatenate(all_embs), all_paths

def train_lora_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
    return running_loss / len(loader)

def build_faiss_index(embs, device):
    faiss_dir = os.path.join(PLANT_HOME, "knn", "faiss_index")
    os.makedirs(faiss_dir, exist_ok=True)
    faiss_file = os.path.join(faiss_dir, "faiss.idx")
    use_gpu = torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
    input_dim = embs.shape[1]
    
    if os.path.exists(faiss_file):
        try:
            index = faiss.read_index(faiss_file)
            if index.d == input_dim:
                if use_gpu:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                return index
        except: pass

    index = faiss.IndexFlatIP(input_dim)
    if use_gpu:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(embs.astype("float32"))
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), faiss_file)
        return gpu_index
    else:
        index.add(embs.astype("float32"))
        faiss.write_index(index, faiss_file)
        return index

def pool_and_aggregate(all_grid_results, min_votes=1):
    quadrat_vote_pool = defaultdict(list)
    print("\nPooling votes from all grids...")
    for grid_idx, (preds, paths) in enumerate(all_grid_results):
        for tile_neighbors, tile_path in zip(preds, paths):
            quadrat_id = os.path.splitext(os.path.basename(tile_path))[0]
            if isinstance(tile_neighbors, np.ndarray):
                votes = tile_neighbors.flatten().tolist()
            else:
                votes = list(tile_neighbors)
            quadrat_vote_pool[quadrat_id].extend(votes)

    print(f"  Aggregating votes for {len(quadrat_vote_pool)} unique quadrats...")
    final_predictions = {}
    for q_id, all_votes in quadrat_vote_pool.items():
        counter = Counter(all_votes)
        species_kept = [sp for sp, count in counter.items() if count >= min_votes]
        final_predictions[q_id] = sorted(species_kept)
    return final_predictions

def main():
    parser = argparse.ArgumentParser(description="PlantCLEF 2025 Pipeline (Multi-GPU)")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "dinov2"])
    parser.add_argument("--quadrat_dir", type=str, required=True)
    parser.add_argument("--single_plant_dir", type=str, required=True)
    parser.add_argument("--grids", type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--votes", type=int, default=3)
    
    # LoRA / Training Args
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to EXISTING LoRA folder")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    BATCH_SIZE = BASE_BATCH_SIZE * num_gpus if num_gpus > 1 else BASE_BATCH_SIZE
    print(f"--- Detected {num_gpus} GPUs. Using Batch Size: {BATCH_SIZE} ---")

    if "dino" in args.backbone.lower(): IMG_SIZE = 518
    else: IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. SETUP DATA ---
    print("\n--- Setting up Single Plant Dataloader ---")
    single_loader_wrapper = SinglePlantDataLoader(
        data_dir=args.single_plant_dir,
        resize_size=IMG_SIZE + 32, img_size=IMG_SIZE,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    train_loader, _, _ = single_loader_wrapper.get_dataloaders()
    train_dataset = train_loader.dataset.dataset 
    class_to_speciesid = {i: int(cls_name) for i, cls_name in enumerate(train_dataset.classes)}
    num_classes = len(class_to_speciesid)

    # --- 2. OPTIONAL LORA TRAINING ---
    lora_path = os.path.join(PLANT_HOME, f"lora_{args.backbone}_best.pth")
    if args.train_lora:
        print("\n--- STARTING LORA TRAINING ---")
        model = get_model(args.backbone, device, num_classes=num_classes, use_lora=True)
        if num_gpus > 1: model = nn.DataParallel(model) 
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs + 1):
            loss = train_lora_epoch(model, train_loader, optimizer, criterion, device, epoch)
            print(f"Epoch {epoch} Loss: {loss:.4f}")
            state = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            torch.save(state, lora_path)
        
        del model
        torch.cuda.empty_cache()
        FORCE_REEXTRACT = True
    else:
        FORCE_REEXTRACT = False

    # --- 3. LOAD MODEL FOR INFERENCE ---
    print("\n--- Loading Model for Inference ---")
    
    if args.lora_checkpoint:
        model = get_model(args.backbone, device, num_classes=None, use_lora=False, lora_checkpoint=args.lora_checkpoint)
    else:
        model = get_model(args.backbone, device, num_classes=None, use_lora=False)

    model = FeatureExtractor(model) 
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # --- 4. EXTRACT EMBEDDINGS ---
    emb_dir = os.path.join(PLANT_HOME, "knn", "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    suffix = ""
    if args.lora_checkpoint: suffix = "_lora_custom"
    elif args.train_lora: suffix = "_lora_trained"
    
    train_emb_file = os.path.join(emb_dir, f"train_embs_{args.backbone}{suffix}.npz")
    if os.path.exists(train_emb_file) and not FORCE_REEXTRACT:
        print(f"Loading cached training embeddings...")
        with np.load(train_emb_file) as data:
            train_embs, train_labels = data['embs'], data['labels']
    else:
        print("Extracting training embeddings...")
        train_embs, train_labels = extract_embeddings_parallel(train_loader, model, device, class_to_speciesid)
        np.savez(train_emb_file, embs=train_embs, labels=train_labels)

    # --- 5. BUILD INDEX ---
    print("Building FAISS Index...")
    faiss.normalize_L2(train_embs)
    index = build_faiss_index(train_embs, device)

    # --- 6. QUADRAT INFERENCE ---
    all_grid_results = [] 
    pil_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for grid_n in args.grids:
        print(f"\n--- Processing Grid Size: {grid_n}x{grid_n} ---")
        quadrat_dataset = QuadratTilingDataset_Inference(
            data_dir=args.quadrat_dir, grid_size=(grid_n, grid_n), transform=pil_transform
        )
        quadrat_loader = DataLoader(
            quadrat_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        quad_emb_file = os.path.join(emb_dir, f"quadrat_embs_{args.backbone}{suffix}_{grid_n}x{grid_n}.npz")
        if os.path.exists(quad_emb_file) and not FORCE_REEXTRACT:
            print(f"Loading cached embeddings...")
            with np.load(quad_emb_file) as data:
                quadrat_embs, quadrat_paths = data['embs'], data['paths']
        else:
            print("Extracting embeddings...")
            quadrat_embs, quadrat_paths = extract_unlabeled_embeddings_parallel(quadrat_loader, model, device)
            np.savez(quad_emb_file, embs=quadrat_embs, paths=quadrat_paths)
            
        print("Running KNN...")
        faiss.normalize_L2(quadrat_embs)
        tile_preds = knn_utils.knn_predict(quadrat_embs, index=index, faiss_labels=train_labels, k=args.neighbors)
        all_grid_results.append((tile_preds, quadrat_paths))

    final_preds = pool_and_aggregate(all_grid_results, min_votes=args.votes)
    out_csv = os.path.join(PLANT_HOME, "submissions", f"submission_{args.backbone}{suffix}.csv")
    knn_utils.write_submission(final_preds, out_csv=out_csv)
    print(f"\nDone! Saved to: {out_csv}")

if __name__ == "__main__":
    main()