import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
from peft import PeftModel, LoraConfig, get_peft_model
import numpy as np
import cv2

# ==========================================
# CONFIGURATION
# ==========================================
#IMAGE_PATH = "/scratch/jme3qd/data/plantclef2025/quadrat/images/2024-CEV3-20240602.jpg" # Path to the specific image you want to visualize
IMAGE_PATH = "/scratch/jme3qd/data/plantclef2025/quadrat/images/CBN-can-A6-20230705.jpg" # Path to the specific image you want to visualize
#IMAGE_PATH = "/scratch/jme3qd/data/plantclef2025/images_max_side_800/1355868/0070793945bc6db2c597387006c5425751204baa.jpg"
OUTPUT_PATH = "./gradcam_quadrat.jpg"
MODEL_TYPE = "baseline" # "lora" or "baseline"
MODEL_CHECKPOINT = "timm/vit_base_patch14_reg4_dinov2.lvd142m"
CLASS_LIST_DIR = "/scratch/jme3qd/data/plantclef2025/images_max_side_800" # Directory to read class names from
GRID_SIZE = (3, 3)
IMAGE_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7806 # Update to match your training
WEIGHTS_PATH = "./lora_lucas_ssl_weights" # <--- POINT THIS TO YOUR TRAINED WEIGHTS FOLDER/FILE
PLANT_HOME = "/scratch/jme3qd/data/plantclef2025"
BACKBONE_PATH = "../dino/baseline_fine_tuned.pth"

# New Configuration Options
SINGLE_PLANT_MODE = False # Set to True for single plant images (disable tiling)

# ==========================================
# GRADCAM CLASS
# ==========================================

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Determine target layer (Last block of the ViT backbone)
        if hasattr(model.backbone, "base_model") and hasattr(model.backbone.base_model, "blocks"):
            self.target_layer = model.backbone.base_model.blocks[-1]
            self.norm_layer = model.backbone.base_model.norm
        elif hasattr(model.backbone, "blocks"):
            self.target_layer = model.backbone.blocks[-1]
            self.norm_layer = model.backbone.norm
        else:
            raise ValueError("Could not find blocks in model backbone")

        # Register Forward Hook
        self.target_layer.register_forward_hook(self.save_activation_and_hook_grad)

    def save_activation_and_hook_grad(self, module, input, output):
        # Save activation
        self.activations = output
        # Register a hook on the tensor itself to catch gradients during backward
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def generate_cam(self, input_tensor, target_class_idx):
        # 1. Zero grads
        self.model.zero_grad()
        
        # 2. Forward Pass
        output = self.model(input_tensor)
        score = output[0, target_class_idx]
        
        # 3. Backward Pass
        score.backward()
        
        # 4. Get Data
        grads = self.gradients
        acts = self.activations
        
        # Debugging: Check if signals are flowing
        if grads is None or acts is None:
            print("Error: No gradients captured. Check if model requires_grad=True for head.")
            return np.zeros((14, 14)) # Dummy return

        # 5. Handle Tokens
        num_patches = (IMAGE_SIZE // 14) ** 2
        offset = acts.shape[1] - num_patches
        
        # Extract patch tokens
        grads = grads[:, offset:, :]
        acts = acts[:, offset:, :]
        
        # 6. Compute Weights (Global Average Pooling of Gradients)
        weights = torch.mean(grads, dim=1, keepdim=True) 
        
        # 7. Weighted Combination
        cam = torch.sum(weights * acts, dim=2) 
        
        # 8. Reshape to Grid
        grid_dim = int(np.sqrt(num_patches))
        cam = cam.reshape(1, grid_dim, grid_dim)
        
        # 9. ReLU and Normalize
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()[0]
        
        # Avoid division by zero
        max_val = np.max(cam)
        min_val = np.min(cam)
        if max_val - min_val > 1e-7:
            cam = (cam - min_val) / (max_val - min_val)
        else:
            cam = np.zeros_like(cam) # Flat heatmap
            
        return cam

# ==========================================
# MODEL DEFINITIONS
# ==========================================

class SimpleClassifier(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.backbone = backbone
        # Dynamic dim check
        if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
             dim = backbone.base_model.model.embed_dim
        elif hasattr(backbone, "embed_dim"):
             dim = backbone.embed_dim
        else:
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
        if hasattr(backbone, "base_model") and hasattr(backbone.base_model, "model"):
             dim = backbone.base_model.model.embed_dim
        else:
             dim = backbone.embed_dim
        self.classifier = nn.Linear(dim, n_classes)
    
    def forward(self, x):
        # Note: We must ensure PEFT forwards properly.
        # Direct call to forward_features might bypass PEFT logic if not careful.
        # But for DINOv2 PEFT usually wraps correctly.
        features = self.backbone.forward_features(x)
        cls_token = features[:, 0]
        logits = self.classifier(cls_token)
        return logits

# ==========================================
# UTILITIES
# ==========================================

def load_class_names(root_dir):
    if os.path.exists(root_dir):
        return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    return [f"Species_{i}" for i in range(NUM_CLASSES)]

def load_model(device, num_classes):
    print(f"Loading {MODEL_TYPE} model...")
    base_model = timm.create_model(MODEL_CHECKPOINT, pretrained=True)
    
    # --- Custom Checkpoint Loading (Challenge Organizers) ---
    if os.path.exists(BACKBONE_PATH):
        print(f"Loading weights from {BACKBONE_PATH}")
        checkpoint = torch.load(BACKBONE_PATH, map_location=device, weights_only=False) 
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        base_model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using downloaded pretrained weights.")
    # -----------------------------------------------------

    if MODEL_TYPE == "lora":
        config = LoraConfig(r=16, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.1, bias="none")
        base_model = get_peft_model(base_model, config)
        
        # Load LoRA Weights
        if os.path.exists(WEIGHTS_PATH):
            print(f"Loading LoRA weights from {WEIGHTS_PATH}...")
            base_model = PeftModel.from_pretrained(base_model, WEIGHTS_PATH)
        else:
             print(f"WARNING: LoRA weights not found at {WEIGHTS_PATH}. Predictions will be random!")

        model = LoRAClassifier(base_model, num_classes)
        
        # Important: The Linear Head is NOT saved in the LoRA adapters folder by default.
        # It's usually saved separately or in a full state_dict.
        # You need to load your classifier head weights here if they aren't in the PEFT folder.
        # Example:
        # full_state = torch.load("final_model_state.pth")
        # model.load_state_dict(full_state)
        
    else:
        model = SimpleClassifier(base_model, num_classes)
        # model.load_state_dict(torch.load("path_to_baseline_checkpoint.pth"))

    model.to(device)
    model.eval() 
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

# ==========================================
# VISUALIZATION
# ==========================================

def apply_heatmap(image_pil, cam_mask):
    w, h = image_pil.size
    cam_mask = cv2.resize(cam_mask, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap)
    result = Image.blend(image_pil.convert("RGB"), heatmap_pil, alpha=0.5)
    return result

def visualize_gradcam(image_path, model, class_names, device):
    if not os.path.exists(image_path):
        print("Image not found.")
        return

    gradcam = ViTGradCAM(model)
    original_img = Image.open(image_path).convert("RGB")
    final_canvas = original_img.copy()
    draw = ImageDraw.Draw(final_canvas)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    # Determine processing mode
    if SINGLE_PLANT_MODE:
        print("Running in SINGLE PLANT MODE (No Tiling)...")
        rows, cols = 1, 1
    else:
        print(f"Running in QUADRAT MODE (Tiling {GRID_SIZE})...")
        rows, cols = GRID_SIZE

    img_width, img_height = original_img.size
    tile_width = img_width // cols
    tile_height = img_height // rows
    
    transform = get_transforms()
    
    print("Generating GradCAM...")
    
    for row in range(rows):
        for col in range(cols):
            left = col * tile_width
            top = row * tile_height
            right = (col + 1) * tile_width if col != cols - 1 else img_width
            bottom = (row + 1) * tile_height if row != rows - 1 else img_height
            
            tile = original_img.crop((left, top, right, bottom))
            input_tensor = transform(tile).unsqueeze(0).to(device)
            input_tensor.requires_grad = True 
            
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            class_id = pred_idx.item()
            
            # Safety check for random models
            if conf.item() < 0.01:
                print(f"Warning: Low confidence ({conf.item():.4f}) on tile {row},{col}. Weights might be random.")

            class_name = class_names[class_id]
            
            cam_map = gradcam.generate_cam(input_tensor, class_id)
            heatmap_tile = apply_heatmap(tile, cam_map)
            final_canvas.paste(heatmap_tile, (left, top))
            
            draw = ImageDraw.Draw(final_canvas) 
            draw.rectangle([left, top, right, bottom], outline="white", width=3)
            
            label_text = f"{class_name}\n{conf.item():.2f}"
            text_pos = (left + 10, top + 10)
            bbox = draw.textbbox(text_pos, label_text, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text(text_pos, label_text, fill="white", font=font)

    final_canvas.save(OUTPUT_PATH)
    print(f"GradCAM visualization saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    classes = load_class_names(CLASS_LIST_DIR)
    model = load_model(DEVICE, len(classes))
    visualize_gradcam(IMAGE_PATH, model, classes, DEVICE)