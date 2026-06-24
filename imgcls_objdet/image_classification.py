
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    "cat", "dog", "bird", "horse", "cow", "sheep", "elephant", "bear",
    "zebra", "giraffe", "lion", "tiger", "rabbit", "fish",
    "person", "baby", "child", "crowd", "athlete", "cyclist",
    "car", "truck", "bus", "motorcycle", "bicycle", "airplane", "boat", "train",
    "chair", "table", "sofa", "laptop", "phone", "bottle", "cup",
    "backpack", "book", "umbrella",
    "pizza", "burger", "cake", "fruit", "vegetable", "coffee",
    "tree", "flower", "mountain", "beach", "ocean", "forest", "snow", "sky",
    "building", "house", "bridge", "skyscraper",
    "sports ball", "tennis racket", "skateboard", "surfboard",
]


def classify(image_path: str, top_k: int = 5):
    print("Loading CLIP  →  openai/clip-vit-base-patch32")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("  ✓ Loaded\n")

    image = Image.open(image_path).convert("RGB")
    print(f"Image  ({image.width}×{image.height})\n")

    # Process image + all labels together in one forward pass
    inputs = processor(
        text=[f"a photo of a {l}" for l in LABELS],
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits_per_image: (1, N_labels) — already scaled cosine similarity
        probs = outputs.logits_per_image.softmax(dim=-1).squeeze(0)  # (N_labels,)

    top_scores, top_idx = probs.topk(top_k)

    # ── Print ─────────────────────────────────────────────────────────────────
    bar_max = 30
    width   = max(len(LABELS[i]) for i in top_idx.tolist()) + 2

    print(f"{'─' * 60}")
    print(f"  {'Label':<{width}}  {'Score':>7}   Confidence")
    print(f"{'─' * 60}")
    for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
        bar   = "█" * int(score * bar_max)
        empty = "░" * (bar_max - len(bar))
        print(f"  {LABELS[idx]:<{width}}  {score:>7.4f}   {bar}{empty}")
    print(f"{'─' * 60}")
    print(f"\nTop prediction: {LABELS[top_idx[0]]}  ({top_scores[0]:.2%} confidence)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--top_k", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classify(args.image, top_k=args.top_k)