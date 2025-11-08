#!/usr/bin/env python
import argparse, os, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

import open_clip
from contextlib import nullcontext

# =========================
# Utilities
# =========================

def set_all_requires_grad(obj, requires_grad: bool):
    """Enable/disable grad on nn.Module, nn.Parameter, iterables, or None."""
    if obj is None:
        return
    if isinstance(obj, nn.Parameter):
        obj.requires_grad = requires_grad
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            set_all_requires_grad(x, requires_grad)
        return
    params = getattr(obj, "parameters", None)
    if callable(params):
        for p in obj.parameters():
            p.requires_grad = requires_grad
        return
    for attr in ("weight", "bias"):
        p = getattr(obj, attr, None)
        if isinstance(p, nn.Parameter):
            p.requires_grad = requires_grad

def get_visual_blocks(visual) -> List[nn.Module]:
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return list(visual.transformer.resblocks)
    if hasattr(visual, "blocks"):
        return list(visual.blocks)
    # fallback: scan for a ModuleList of residual attention blocks
    for m in visual.modules():
        if isinstance(m, nn.ModuleList) and len(m) > 0 and "Residual" in m[0].__class__.__name__:
            return list(m)
    raise RuntimeError("Could not locate visual transformer blocks.")

def get_autocast_and_scaler(device_type: str, use_amp: bool):
    if not use_amp:
        return nullcontext(), None
    if device_type == "cuda":
        return torch.cuda.amp.autocast(), torch.cuda.amp.GradScaler()
    if device_type == "mps":
        # GradScaler not needed on MPS; autocast is available
        return torch.autocast(device_type="mps", dtype=torch.float16), None
    return nullcontext(), None

def cosine_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# =========================
# TinyImageNet helpers
# =========================

def read_wnids(data_root: Path) -> List[str]:
    wnids_path = data_root / "wnids.txt"
    if not wnids_path.exists():
        raise FileNotFoundError(f"Missing {wnids_path}")
    return [ln.strip() for ln in wnids_path.read_text().splitlines() if ln.strip()]

class TinyImageNetValFromAnnotations(Dataset):
    """Original TinyImageNet val/ layout reader using val_annotations.txt."""
    def __init__(self, data_root: Path, preprocess, wnids: List[str]):
        self.root = Path(data_root)
        self.preprocess = preprocess
        self.wnids = wnids
        self.wnid_to_idx = {w: i for i, w in enumerate(wnids)}

        ann = self.root / "val" / "val_annotations.txt"
        img_dir = self.root / "val" / "images"
        if not ann.exists():
            raise FileNotFoundError(f"Missing {ann}")
        self.samples: List[Tuple[Path, int]] = []
        for line in ann.read_text().strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(" ")
            img_name, wnid = parts[0], parts[1]
            if wnid not in self.wnid_to_idx:
                continue
            self.samples.append((img_dir / img_name, self.wnid_to_idx[wnid]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
        return self.preprocess(img), y

def remap_imagefolder_to_canonical(ds: datasets.ImageFolder, canonical_classes: List[str]):
    """Force ImageFolder dataset to use canonical index order defined by wnids.txt."""
    from pathlib import Path as _Path
    canonical = {w: i for i, w in enumerate(canonical_classes)}
    new_samples = []
    for path, _y in ds.samples:
        p = _Path(path)
        # TinyImageNet train layout nests files in train/<wnid>/images/*.JPEG
        while p.name not in canonical and p.parent != p:
            p = p.parent
        cname = p.name
        if cname not in canonical:
            raise KeyError(
                f"Sample {path} is not inside any known wnid directory from wnids.txt."
            )
        new_samples.append((path, canonical[cname]))
    ds.samples = new_samples
    ds.targets = [y for _, y in new_samples]
    ds.class_to_idx = canonical
    ds.classes = list(canonical_classes)

# =========================
# Model head
# =========================

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing) if label_smoothing > 0 else nn.CrossEntropyLoss()

    def forward(self, x): return self.fc(x)

# =========================
# Train / Eval
# =========================

def train_one_epoch(model, head, loader, optimizer, scaler, device, epoch, autocast_ctx, accumulation_steps=1):
    model.train(); head.train()
    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with autocast_ctx:
            feats = model.encode_image(images)         # [B, D], L2-normalized
            logits = head(feats)
            loss = head.loss(logits, targets) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    return running_loss / len(loader), 100.0 * correct / total

@torch.no_grad()
def evaluate(model, head, loader, device, autocast_ctx):
    model.eval(); head.eval()
    loss_fn = head.loss
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast_ctx:
            feats = model.encode_image(images)
            logits = head(feats)
            loss = loss_fn(logits, targets)
        running_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    return running_loss / len(loader), 100.0 * correct / total

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="tiny-imagenet-200", help="Path to tiny-imagenet-200")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr_head", type=float, default=5e-4, help="LR for linear head")
    ap.add_argument("--lr_backbone", type=float, default=1e-4, help="LR for unfrozen CLIP params")
    ap.add_argument("--wd_head", type=float, default=0.05)
    ap.add_argument("--wd_backbone", type=float, default=0.1)
    ap.add_argument("--unfreeze_k", type=int, default=2, help="Unfreeze last K visual transformer blocks")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--val_from_annotations", action="store_true",
                    help="Read val/ from val_annotations.txt instead of ImageFolder")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--warmup_epochs", type=int, default=5)
    args = ap.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Model + preprocessors
    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model.to(device)

    # Freeze everything
    set_all_requires_grad(model, False)

    # Unfreeze last K visual blocks + ln_post/proj
    visual = model.visual
    blocks = get_visual_blocks(visual)
    k = max(0, min(args.unfreeze_k, len(blocks)))
    if k > 0:
        for b in blocks[-k:]:
            set_all_requires_grad(b, True)
        print(f"Unfroze last {k} visual transformer blocks out of {len(blocks)}.")
    else:
        print("Keeping all visual transformer blocks frozen.")

    # Always allow final norms/projection to learn
    if hasattr(visual, "ln_post"):
        set_all_requires_grad(visual.ln_post, True)
    proj = getattr(visual, "proj", None)
    if isinstance(proj, nn.Parameter):
        proj.requires_grad = True
    else:
        set_all_requires_grad(proj, True)

    # Datasets with canonical class mapping from wnids.txt
    root = Path(args.data_root)
    wnids = read_wnids(root)

    # Train dataset (ImageFolder) + remap to canonical order
    train_ds = datasets.ImageFolder(root / "train", transform=train_preprocess)
    # Sanity: ensure train contains all wnids
    missing = set(wnids) - set(train_ds.classes)
    if missing:
        raise RuntimeError(f"Train set missing classes: {sorted(list(missing))[:5]} ...")
    remap_imagefolder_to_canonical(train_ds, wnids)

    # Val dataset: either original annotations or reorganized ImageFolder
    if args.val_from_annotations:
        val_ds = TinyImageNetValFromAnnotations(root, val_preprocess, wnids)
    else:
        val_ds = datasets.ImageFolder(root / "val", transform=val_preprocess)
        # If you reorganized val into per-class folders, remap to canonical order
        if len(val_ds.classes) == 200:
            remap_imagefolder_to_canonical(val_ds, wnids)
        else:
            raise RuntimeError(
                "val/ does not look like an ImageFolder with 200 class dirs. "
                "Use --val_from_annotations to read the original layout."
            )

    num_classes = 200
    assert len(train_ds.classes) == num_classes

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device == "cuda")
    )

    # Embedding dim detection
    if hasattr(model.visual, "output_dim"):
        embed_dim = model.visual.output_dim
    elif isinstance(getattr(model.visual, "proj", None), nn.Parameter):
        embed_dim = model.visual.proj.shape[1]
    elif hasattr(model.visual, "proj"):
        embed_dim = model.visual.proj.out_features
    else:
        embed_dim = model.text_projection.shape[1]

    head = LinearHead(embed_dim, num_classes, label_smoothing=args.label_smoothing).to(device)

    # Optimizer with param groups
    def trainable_params(m):
        return [p for p in m.parameters() if p.requires_grad]

    backbone_params = trainable_params(model)
    param_groups = [
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": args.wd_backbone},
        {"params": head.parameters(), "lr": args.lr_head, "weight_decay": args.wd_head},
    ]
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(1, len(train_loader) // max(1, args.accum_steps))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch if args.cosine else 0
    scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps) if args.cosine else None

    autocast_ctx, scaler = get_autocast_and_scaler(device, args.fp16)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, head, train_loader, optimizer, scaler, device, epoch, autocast_ctx, args.accum_steps
        )
        val_loss, val_acc = evaluate(model, head, val_loader, device, autocast_ctx)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Epoch {epoch} | Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
                "classes": train_ds.classes,
                "args": vars(args),
            }
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(ckpt, f"checkpoints/openclip_{args.model.replace('/','-')}_tinyimg_k{args.unfreeze_k}_best.pt")
            print(f"Saved new best checkpoint (acc={best_acc:.2f}%).")

    print(f"Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
