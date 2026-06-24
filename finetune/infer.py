
import glob
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import transformers

from train import DinoV3Linear

CKPT_PATH = "./weights/model_best.pt"
DATA_DIR = "./CUB_200_2011/images"  # used here only to grab a sample image to test on


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    proc_cfg = dict(ckpt["config"]["image_processor"])
    processor_type = proc_cfg.pop("image_processor_type")
    ProcessorClass = getattr(transformers, processor_type)
    image_processor = ProcessorClass(**proc_cfg)

    backbone_cfg = ckpt["config"]["backbone"]
    auto_config = transformers.AutoConfig.for_model(**backbone_cfg)
    backbone = transformers.AutoModel.from_config(auto_config)

    model = DinoV3Linear(
        backbone=backbone,
        num_classes=len(ckpt["config"]["classes"]),
        freeze_backbone=ckpt["config"].get("freeze_backbone", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    classes = ckpt["config"]["classes"]
    return model, image_processor, classes


@torch.no_grad()
def infer(model, image_processor, classes, image, device):
    image = image.convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    logits = model(pixel_values)
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(dim=-1).item()
    conf = probs[0, pred_idx].item()
    pred_class = classes[pred_idx]
    return pred_class, conf


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, image_processor, classes = load_model(CKPT_PATH, device)

    # Grab a random sample image from the dataset directory to sanity-check inference
    images = glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)
    if not images:
        raise FileNotFoundError(f"No .jpg images found under {DATA_DIR}")

    image_path = np.random.choice(images)
    image = Image.open(image_path)

    pred, conf = infer(model, image_processor, classes, image, device)
    print(f"Image: {image_path}")
    print(f"Predicted: {pred}, Conf: {conf:.4f}")