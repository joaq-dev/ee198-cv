
import json
import math
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup
import trackio


BATCH_SIZE = 128          # Adjust based on your GPU VRAM
LR = 5e-3                # Learning rate for the AdamW optimizer
WEIGHT_DECAY = 0.01      # Weight decay for regularization
EPOCHS = 20               # Numbr of training loops
WARMUP_RATIO = 0.1       # 10% of training steps used for linear warmup

DATA_DIR = "./CUB_200_2011/images"
CKPT_DIR = "./weights"
CKPT_PATH = os.path.join(CKPT_DIR, "model_best.pt")
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

NUM_WORKERS = 4  



class DinoV3Linear(nn.Module):
    def __init__(self, backbone: AutoModel, num_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden_size = getattr(backbone.config, "hidden_size", None)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.bacekbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0]
        logits = self.head(cls)
        return logits



_image_processor_cache = {}


def get_image_processor():
    """Lazily build/cache the image processor per-process (each DataLoader worker
    process gets its own cache entry, avoiding repeated reconstruction per-sample)."""
    if "processor" not in _image_processor_cache:
        _image_processor_cache["processor"] = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return _image_processor_cache["processor"]


def dino_transform(img):
    # ImageFolder can hand back non-RGB images (grayscale/CMYK/RGBA); normalize to RGB
    img = img.convert("RGB")
    processor = get_image_processor()
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"][0]
    return pixel_values


def save_checkpoint(path, model, optimizer, scheduler, full_dataset, backbone_config,
                     image_processor_config, freeze_backbone, global_step, epoch, acc):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "model_name": MODEL_NAME,
                "classes": full_dataset.classes,
                "backbone": backbone_config,
                "image_processor": image_processor_config,
                "freeze_backbone": freeze_backbone,
            },
            "step": global_step,
            "epoch": epoch,
            "val_acc": acc,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for pixel_values, labels in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load processor + backbone
    image_processor = get_image_processor()
    backbone = AutoModel.from_pretrained(MODEL_NAME)

    image_processor_config = json.loads(image_processor.to_json_string())
    backbone_config = json.loads(AutoConfig.from_pretrained(MODEL_NAME).to_json_string())

    # Dataset / dataloaders
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=dino_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    num_classes = len(full_dataset.classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,  # avoids respawning workers every epoch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    # Model
    freeze_backbone = True
    model = DinoV3Linear(backbone, num_classes, freeze_backbone=freeze_backbone).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = EPOCHS * math.ceil(len(train_loader))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    global_step = 0

    trackio.init(
        project="dinov3",
        config={
            "epochs": EPOCHS,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
        },
    )

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        if freeze_backbone:
            model.backbone.eval()  # keep frozen backbone in eval mode

        running_loss = 0.0
        for i, (pixel_values, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(pixel_values)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 10 == 0:
                trackio.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        trackio.log(
            {
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "val/loss": val_loss,
                "val/acc": val_acc,
            },
            step=global_step,
        )

        # Save "last" checkpoint every epoch, and "best" only when val accuracy improves
        save_checkpoint(
            os.path.join(CKPT_DIR, "model_last.pt"), model, optimizer, scheduler,
            full_dataset, backbone_config, image_processor_config, freeze_backbone,
            global_step, epoch, val_acc,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                CKPT_PATH, model, optimizer, scheduler, full_dataset, backbone_config,
                image_processor_config, freeze_backbone, global_step, epoch, val_acc,
            )
            print(f"  -> new best (val_acc={val_acc:.4f}), saved to {CKPT_PATH}")

    print(f"Training complete. Best val_acc={best_acc:.4f}")
    trackio.finish()


if __name__ == "__main__":
    main()