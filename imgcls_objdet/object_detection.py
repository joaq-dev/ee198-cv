
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy import ndimage

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DINO_REPO   = "./dinov3-main"
DINO_MODEL  = "dinov3_vits16"
WEIGHTS     = "./weights/dinov3_vits16.pth"
PATCH_SIZE  = 16
IMG_SIZE    = 448

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # squash to square for model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_backbone(repo: str, model: str, weights: str) -> torch.nn.Module:
    print(f"Loading DINOv3 from  →  {repo}")
    backbone = torch.hub.load(repo, model, source="local", weights=weights)
    backbone = backbone.to(DEVICE).eval()
    print("  ✓ Backbone loaded\n")
    return backbone

def get_attention_map(backbone: torch.nn.Module, tensor: torch.Tensor, ph: int, pw: int) -> np.ndarray:
    """Uses raw DINOv3 feature similarity relative to a target center patch."""
    feat_store = {}
    def hook(module, input, output):
        feat_store["feat"] = output.detach()
        
    handle = backbone.norm.register_forward_hook(hook)
    with torch.no_grad():
        _ = backbone(tensor)
    handle.remove()
    
    if "feat" not in feat_store:
        raise RuntimeError("Hook did not fire — backbone.norm not found.")
        
    feat = feat_store["feat"] # (1, seq_len, dim)
    num_patches = ph * pw
    patch_feats = feat[0, -num_patches:, :] # Pure DINOv3 dense patch features
    
    # Pick the absolute center patch as our query anchor
    center_idx = (ph // 2) * pw + (pw // 2)
    anchor_feature = patch_feats[center_idx : center_idx + 1, :] # Shape (1, dim)
    
    # Calculate Cosine Similarity purely using DINOv3's native embeddings
    norm_patches = patch_feats / (patch_feats.norm(dim=-1, keepdim=True) + 1e-8)
    norm_anchor = anchor_feature / (anchor_feature.norm(dim=-1, keepdim=True) + 1e-8)
    
    similarity = torch.mm(norm_patches, norm_anchor.T).squeeze(-1) # Shape: (784)
    
    # Map back to 2D image coordinates
    attn_map = similarity.cpu().numpy().reshape(ph, pw)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    return attn_map

def attn_to_boxes(attn_map: np.ndarray, threshold: float, orig_w: int, orig_h: int) -> list[dict]:
    """Threshold similarity map → connected blobs → bounding boxes."""
    ph, pw = attn_map.shape
    try:
        from skimage.filters import threshold_otsu
        auto_thresh = threshold_otsu(attn_map)
        print(f"Otsu threshold: {auto_thresh:.4f}")
        binary = (attn_map > auto_thresh).astype(np.uint8)
    except ImportError:
        binary = (attn_map > threshold).astype(np.uint8)
        
    labeled, n = ndimage.label(binary)
    boxes = []
    for i in range(1, n + 1):
        region = np.where(labeled == i)
        if len(region[0]) < 2:
            continue
        r_min, r_max = region[0].min(), region[0].max() + 1
        c_min, c_max = region[1].min(), region[1].max() + 1
        x1 = int(c_min / pw * orig_w)
        y1 = int(r_min / ph * orig_h)
        x2 = int(c_max / pw * orig_w)
        y2 = int(r_max / ph * orig_h)
        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "score": float(attn_map[region].mean()),
        })
    return sorted(boxes, key=lambda b: b["score"], reverse=True)

def detect_and_draw(image_path: str, threshold: float = 0.6, output: str = "output.jpg"):
    backbone      = load_backbone(DINO_REPO, DINO_MODEL, WEIGHTS)
    image         = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.width, image.height
    print(f"Image  ({orig_w}×{orig_h})\n")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    ph = pw = IMG_SIZE // PATCH_SIZE    # 28 × 28
    
    attn_map = get_attention_map(backbone, tensor, ph, pw)
    
    # Scale map to original image size
    from scipy.ndimage import zoom
    heatmap = zoom(attn_map, (orig_h / ph, orig_w / pw))
    boxes = attn_to_boxes(attn_map, threshold=threshold, orig_w=orig_w, orig_h=orig_h)
    print(f"Found {len(boxes)} region(s)\n")
    
    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("DINOv3 ViT-S/16  |  Native Feature Similarity Detection", fontsize=13, fontweight="bold")
    
    axes[0].imshow(image)
    axes[0].imshow(heatmap, cmap="inferno", alpha=0.55)
    axes[0].set_title("DINOv3 Feature Similarity Heatmap")
    axes[0].axis("off")
    
    axes[1].imshow(image)
    axes[1].set_title(f"Detected Regions  ({len(boxes)} found)")
    axes[1].axis("off")
    
    cmap = plt.get_cmap("tab10")
    for i, box in enumerate(boxes):
        color = cmap(i % 10)
        axes[1].add_patch(patches.Rectangle(
            (box["x1"], box["y1"]),
            box["x2"] - box["x1"],
            box["y2"] - box["y1"],
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        axes[1].text(
            box["x1"], box["y1"] - 6,
            f"region {i+1}  {box['score']:.2f}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=3, edgecolor="none"),
        )
        print(f"  Region {i+1}  score={box['score']:.4f}  "
              f"box=({box['x1']},{box['y1']},{box['x2']},{box['y2']})")
              
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved  →  {output}")
    plt.show()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",     required=True)
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Similarity threshold 0–1  (lower = more boxes, default: 0.6). "
                        "Ignored if scikit-image is installed (Otsu auto-threshold used).")
    p.add_argument("--output",    default="output.jpg")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    detect_and_draw(args.image, threshold=args.threshold, output=args.output)