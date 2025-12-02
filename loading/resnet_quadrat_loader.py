import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import glob
from pathlib import Path

def tile_image_nxn(img, tiles_per_side=1, target_size=224):  # no tiling for the inital Baseline 1
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

class QuadratNxNDataset(Dataset):
    def __init__(self, root, transform, tiles_per_side=1, target_size=224):
        self.paths = sorted([str(p) for p in Path(root).glob("*.*")])
        self.transform = transform
        self.tiles_per_side = tiles_per_side
        self.target_size = target_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        # This portion centercrops the image into a square image
        # width, height = img.size
        # short_side = min(width, height)
        # center_crop_transform = transforms.CenterCrop(short_side)
        # img = center_crop_transform(img)

        # Generating tiles
        tiles = tile_image_nxn(
            img,
            tiles_per_side=self.tiles_per_side,
            target_size=self.target_size
            )

        tiles = [self.transform(t) for t in tiles]
        tiles = torch.stack(tiles)  # shape [N*N, 3, 224, 224]

        return tiles, path

# Preprocess the test/quadrat data
def main(inference_data_path, tiles_per_side=1, target_size=224, batch_size=32, num_workers=4):
    print(f"Loading data from: {inference_data_path}")
    try:
        inference_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)), 
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
                ])
        
        inference_data = QuadratNxNDataset(
                    root=inference_data_path,
                    transform=inference_transform,
                    tiles_per_side=tiles_per_side,
                    target_size=target_size
                    )
        
        inference_loader = DataLoader(
                inference_data,
                batch_size,
                shuffle=False,   # critical for inference to keep tiles in order
                num_workers=num_workers
            )
        print("\n--- Dataloader Pipelines Created ---")
        
    except Exception as e:
        print(f"An unexpected error occurred loading data: {e}")

    return inference_data, inference_loader

if __name__ == "__main__":
    main()

