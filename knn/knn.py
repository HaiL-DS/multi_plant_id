import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
import pandas as pd
import os
import faiss
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Subset
from collections import defaultdict
from sklearn.metrics import silhouette_score
import csv
from pathlib import Path

from matplotlib import pyplot as plt
import random
from PIL import Image

def tile_image_nxn(img, tiles_per_side=3, target_size=518):
    w, h = img.size

    tile_w = w / tiles_per_side
    tile_h = h / tiles_per_side

    tiles = []

    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            left = int(col * tile_w)
            top = int(row * tile_h)
            right = int((col + 1) * tile_w)
            bottom = int((row + 1) * tile_h)

            tile = img.crop((left, top, right, bottom))
            tile = tile.resize((target_size, target_size), Image.BICUBIC)
            tiles.append(tile)

    return tiles


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

class QuadratNxNDataset(Dataset):
    def __init__(self, root, transform, tiles_per_side=3, target_size=518):
        self.paths = sorted([str(p) for p in Path(root).glob("*.*")])
        self.transform = transform
        self.tiles_per_side = tiles_per_side
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        tiles = tile_image_nxn(
            img,
            tiles_per_side=self.tiles_per_side,
            target_size=self.target_size
        )

        tiles = [self.transform(t) for t in tiles]
        tiles = torch.stack(tiles)  # shape [N*N, 3, 518, 518]

        return tiles, path


train_dataset = datasets.ImageFolder("/scratch/jme3qd/data/plantclef2025/images_max_side_800", transform=transform)

n_samples = 20000
indices = np.random.choice(len(train_dataset), n_samples, replace=False)
train_subset = Subset(train_dataset, indices)

# %%
#train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)


from pathlib import Path
quadrat_path = "/scratch/jme3qd/data/plantclef2025/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images"

quadrat_loader = DataLoader(
    QuadratNxNDataset(
        quadrat_path,
        transform,
        tiles_per_side=3,   # <--- YOU CONTROL TILE COUNT HERE
        target_size=518
    ),
    batch_size=1,
    shuffle=False,
    num_workers=4
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model("timm/vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)
#model = timm.create_model("vit_base_patch16_224", pretrained=True)
checkpoint_path = '/home/jme3qd/Downloads/model_best.pth.tar'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # Load to CPU first

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint # Assume the checkpoint itself is the state_dict

# 3. Load the state dictionary into the model
model.load_state_dict(state_dict,strict=False)

model.eval().to(device)
for p in model.parameters():
    p.requires_grad = False


def extract_embeddings(dataloader):
    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            if feats.ndim == 3:
                cls_embs = feats[:, 0, :]  # [CLS] token
            else:
                cls_embs = feats
            all_embs.append(cls_embs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_embs), np.concatenate(all_labels)

train_file = "embs_train.npz"
if os.path.exists(train_file):
    with np.load(train_file) as data:
        train_embs = data['embs']
        train_labels = data['labels']
        print("Train embeddings loaded from", train_file)
else:
    train_embs, train_labels = extract_embeddings(train_loader)
    np.savez(train_file, embs=train_embs, labels=train_labels)



def extract_unlabeled_embeddings(dataloader, model, device):
    all_embs, all_paths = [], []
    with torch.no_grad():
        for imgs, paths in tqdm(dataloader):
            imgs = imgs.to(device)
            feats = model.forward_features(imgs)
            cls_embs = feats[:, 0, :] if feats.ndim == 3 else feats
            all_embs.append(cls_embs.cpu().numpy())
            all_paths.extend(paths)
    return np.concatenate(all_embs), all_paths

filename = "quadrat_embs.npz"
if os.path.exists(filename):
    with np.load(filename) as data:
        quadrat_embs = data['embs']
        quadrat_paths = data['paths']
        print("Quadrat embeddings loaded from", filename)
else:
    quadrat_embs, quadrat_paths = extract_unlabeled_embeddings(quadrat_loader, model, device)
    np.savez(filename, embs=quadrat_embs, paths=quadrat_paths)


faiss.normalize_L2(quadrat_embs)
faiss.normalize_L2(train_embs)
index = faiss.IndexFlatIP(train_embs.shape[1])
index.add(train_embs)
D, I = index.search(quadrat_embs, k=5)

quadrat_tile_predictions = defaultdict(list)

with torch.no_grad():
    for tiles, path in tqdm(quadrat_loader):
        path = path[0]
        tiles = tiles.to(device)

        T = tiles.shape[1]
        tiles = tiles.view(T, 3, 518, 518)

        feats = model.forward_features(tiles)
        cls = feats[:, 0, :].cpu().numpy()

        k = 5
        distances, indices = index.search(cls, k)

        tile_labels = train_labels[indices]
        quadrat_tile_predictions[path] = tile_labels


quadrat_final_predictions = {}

for path, tile_knn_labels in quadrat_tile_predictions.items():
    flat = tile_knn_labels.flatten()
    counts = Counter(flat)

    total_votes = len(flat)
    threshold = 0.05 * total_votes

    preds = [s for s, c in counts.items() if c >= threshold]
    quadrat_final_predictions[path] = preds

with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["quadrat_id", "species_ids"])

    for path, species_list in quadrat_final_predictions.items():
        quadrat_id = Path(path).stem  # filename without extension

        # Convert list → "[1, 2, 3]"
        species_str = "[" + ", ".join(str(s) for s in species_list) + "]"

        writer.writerow([quadrat_id, species_str])



sil_score = silhouette_score(train_embs, train_labels, metric='cosine')
print(f"Silhouette score: {sil_score:.3f}")



D_train, I_train = index.search(train_embs, k=5)
purity = np.mean([
    np.mean(train_labels[i] == train_labels[nbrs]) 
    for i, nbrs in enumerate(I_train)
])
print(f"Neighborhood purity (k=5): {purity:.3f}")

k = 5
idx = random.randint(0, len(quadrat_paths)-1)
nbrs = I[idx]
fig, axes = plt.subplots(1, k+1, figsize=(15, 4))
axes[0].imshow(Image.open(quadrat_paths[idx]))
axes[0].set_title("Quadrat tile")
for j, n in enumerate(nbrs):
    axes[j+1].imshow(Image.open(train_dataset.samples[n][0]))
    axes[j+1].set_title(train_dataset.classes[train_labels[n]])
plt.savefig(f"quadrat_{idx}.png")