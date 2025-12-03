import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
#IMAGE_PATH = "/scratch/jme3qd/data/plantclef2025/quadrat/images/2024-CEV3-20240602.jpg"
IMAGE_PATH = "/scratch/jme3qd/data/plantclef2025/quadrat/images/CBN-can-A6-20230705.jpg" # Path to the specific image you want to visualize
OUTPUT_PATH = "./annotated_quadrat.jpg"
MODEL_TYPE = "baseline" # Options: "lora" or "baseline"
NUM_CLASSES = 7806 # Update this to match your actual number of classes
CLASS_LIST_DIR = "/scratch/jme3qd/data/plantclef2025/images_max_side_800" # Directory to read class names from
MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
GRID_SIZE = (3, 3) # Rows, Cols
IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# MODEL DEFINITIONS (Must match training)
# ==========================================

class SimpleClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        # Handle cases where backbone might be a PeftModel or standard timm model
        if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
             dim = backbone.base_model.model.embed_dim
        elif hasattr(backbone, "embed_dim"):
             dim = backbone.embed_dim
        else:
             # Fallback for some timm structures
             dim = backbone.num_features
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

class LoRAClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        # Detect embedding dim dynamically
        if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
             dim = backbone.base_model.model.embed_dim
        else:
             dim = backbone.embed_dim
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# UTILITIES
# ==========================================

def load_class_names(root_dir):
    """Loads class names from the directory structure."""
    if os.path.exists(root_dir):
        return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    else:
        print(f"Warning: Class directory {root_dir} not found. Using dummy IDs.")
        return [f"Species_{i}" for i in range(NUM_CLASSES)]

def load_model(device, num_classes):
    print(f"Loading {MODEL_TYPE} model...")
    
    # 1. Load Base Backbone (timm)
    base_model = timm.create_model(MODEL_CHECKPOINT, pretrained=True)
    
    if MODEL_TYPE == "lora":
        # Inject LoRA Config
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        base_model = get_peft_model(base_model, config)
        
        # Wrap in Classifier
        model = LoRAClassifier(base_model, num_classes)
        
        # Load Weights (If you saved the full state dict)
        # model.load_state_dict(torch.load("path_to_lora_checkpoint.pth"))
        
    else:
        # Baseline
        model = SimpleClassifier(base_model, num_classes)
        # Load Weights
        # model.load_state_dict(torch.load("path_to_baseline_checkpoint.pth"))

    model.to(device)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Add Normalization if used during training
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# ==========================================
# VISUALIZATION LOGIC
# ==========================================

def visualize_prediction(image_path, model, class_names, device):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Load original image
    original_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    img_width, img_height = original_img.size
    rows, cols = GRID_SIZE
    
    tile_width = img_width // cols
    tile_height = img_height // rows
    
    transform = get_transforms()
    
    print("Processing tiles...")
    
    for row in range(rows):
        for col in range(cols):
            # Calculate coordinates
            left = col * tile_width
            top = row * tile_height
            right = (col + 1) * tile_width if col != cols - 1 else img_width
            bottom = (row + 1) * tile_height if row != rows - 1 else img_height
            
            # Crop
            tile = original_img.crop((left, top, right, bottom))
            
            # Prepare for model
            input_tensor = transform(tile).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                
            class_name = class_names[pred_idx.item()]
            confidence = conf.item()
            
            # --- Draw on Image ---
            
            # Draw Box (Red)
            draw.rectangle([left, top, right, bottom], outline="red", width=5)
            
            # Draw Label (White text with Black background for readability)
            label_text = f"{class_name}\n({confidence:.2f})"
            
            # Calculate text position (top left of tile with padding)
            text_pos = (left + 10, top + 10)
            
            # Draw text background
            bbox = draw.textbbox(text_pos, label_text, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text(text_pos, label_text, fill="white", font=font)

    # Save result
    original_img.save(OUTPUT_PATH)
    print(f"Visualization saved to {OUTPUT_PATH}")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # 1. Load Classes
    classes = load_class_names(CLASS_LIST_DIR)
    
    # 2. Load Model
    model = load_model(DEVICE, len(classes))
    
    # 3. Run Visualization
    visualize_prediction(IMAGE_PATH, model, classes, DEVICE)