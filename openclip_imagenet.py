#!/usr/bin/env python
"""
Zero‑shot classification on Tiny ImageNet using OpenCLIP with ViT‑B/32 and ViT‑B/16

- Loads strong LAION checkpoints (defaults: ViT‑B/32: laion2b_s34b_b79k, ViT‑B/16: laion2b_s34b_b88k)
- Builds zero‑shot text embeddings from Tiny ImageNet class names (words.txt)
- Evaluates Top‑1 and Top‑5 accuracy on the validation set

Usage:
    python zero_shot_tinyimagenet_openclip.py \
        --data-root /path/to/tiny-imagenet-200 \
        --batch-size 256 --num-workers 8 --device cuda

You may select specific models or checkpoints, e.g.:
    --model ViT-B-32 --pretrained laion2b_s34b_b79k
    --model ViT-B-16 --pretrained laion2b_s34b_b88k

Tiny ImageNet expected layout:
    tiny-imagenet-200/
      ├── wnids.txt
      ├── words.txt
      ├── train/
      └── val/
          ├── images/
          └── val_annotations.txt

"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import open_clip

# ---------------------------
# Utilities
# ---------------------------

def read_wnids_words(data_root: Path) -> Tuple[List[str], Dict[str, str]]:
    """Reads wnids.txt and words.txt -> (ordered wnids list, mapping wnid -> human label).
    words.txt lines look like: n01443537\ttench, Tinca tinca
    """
    wnids_path = data_root / "wnids.txt"
    words_path = data_root / "words.txt"
    if not wnids_path.exists() or not words_path.exists():
        raise FileNotFoundError("wnids.txt or words.txt not found under the dataset root.")

    wnids = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
    wmap: Dict[str, str] = {}
    for line in words_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            wnid, names = line.split("\t", 1)
            # Use the first synonym as the primary label; keep others for prompts
            primary = names.split(",")[0].strip()
            wmap[wnid] = primary
        except ValueError:
            continue
    # Fallback: if a wnid has no words entry, use wnid itself
    for w in wnids:
        wmap.setdefault(w, w)
    return wnids, wmap

# Prompt templates adapted from CLIP zero-shot evaluation
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
    "a jpeg corrupted photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a low resolution photo of a {}.",
]

class TinyImageNetVal(Dataset):
    """Validation split for Tiny ImageNet using val_annotations.txt.
    Returns (image_tensor, class_index).
    """
    def __init__(self, data_root: Path, preprocess, wnids: List[str]):
        self.data_root = Path(data_root)
        self.preprocess = preprocess
        self.wnids = wnids
        self.val_img_dir = self.data_root / "val" / "images"
        ann_path = self.data_root / "val" / "val_annotations.txt"
        if not ann_path.exists():
            raise FileNotFoundError("val/val_annotations.txt not found.")
        self.samples: List[Tuple[Path, int]] = []
        content = ann_path.read_text().strip().splitlines()
        wnid_to_idx = {w: i for i, w in enumerate(self.wnids)}
        for line in content:
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(" ")  # some dists are space-separated
            img_name = parts[0]
            wnid = parts[1]
            img_path = self.val_img_dir / img_name
            if wnid not in wnid_to_idx:
                # skip if label isn't in wnids.txt (shouldn't happen)
                continue
            self.samples.append((img_path, wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img = self.preprocess(img)
        return img, target

@torch.no_grad()
def build_text_features(model, tokenizer, device, classnames: List[str], templates: List[str]) -> torch.Tensor:
    """Build normalized text features by averaging over prompt templates per class.
    Returns tensor of shape [num_classes, dim].
    """
    texts = []
    for cname in classnames:
        prompts = [t.format(cname) for t in templates]
        texts.append(tokenizer(prompts))
    # texts: list of [num_prompts, context_length]
    text_embeds = []
    for t in texts:
        t = t.to(device)
        class_embeds = model.encode_text(t)
        class_embeds = F.normalize(class_embeds, dim=-1)
        class_embed = class_embeds.mean(dim=0)
        class_embed = F.normalize(class_embed, dim=-1)
        text_embeds.append(class_embed)
    text_features = torch.stack(text_embeds, dim=0)
    return text_features

@torch.no_grad()
def evaluate(model, dataloader: DataLoader, text_features: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    top1 = 0
    top5 = 0
    logit_scale = model.logit_scale.exp()
    text_features = text_features.to(device)

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        # Top-k
        _, pred = logits.topk(5, dim=-1)
        top1 += (pred[:, 0] == targets).sum().item()
        # For top-5, check membership
        in_top5 = (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        top5 += in_top5
        total += targets.size(0)

    return top1 / total, top5 / total


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot Tiny ImageNet with OpenCLIP")
    p.add_argument("--data-root", type=str, default="tiny-imagenet-200", help="Path to tiny-imagenet-200 root directory")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Model selection: can pass multiple --model/--pretrained pairs; if not provided, run both defaults
    p.add_argument("--model", action="append", default=[], help="Model name, e.g., ViT-B-32 or ViT-B-16")
    p.add_argument("--pretrained", action="append", default=[], help="Pretrained tag from open_clip (same length/order as --model)")

    # Mixed precision / performance knobs
    p.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for image encoding")
    p.add_argument("--templates", type=int, default=len(TEMPLATES), help="Number of prompt templates to use (<= %d)" % len(TEMPLATES))
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    # Defaults if user didn't specify models
    model_specs: List[Tuple[str, str]] = []
    if not args.model:
        model_specs = [
            ("ViT-B-32", "laion2b_s34b_b79k"),
            ("ViT-B-16", "laion2b_s34b_b88k"),
        ]
    else:
        if args.pretrained and len(args.pretrained) != len(args.model):
            raise ValueError("If --pretrained is provided, it must match the number of --model entries.")
        for i, m in enumerate(args.model):
            pt = args.pretrained[i] if i < len(args.pretrained) else None
            if pt is None:
                # Sensible defaults per architecture
                if m == "ViT-B-32":
                    pt = "laion2b_s34b_b79k"
                elif m == "ViT-B-16":
                    pt = "laion2b_s34b_b88k"
                else:
                    pt = "laion2b_s34b_b79k"  # generic default
            model_specs.append((m, pt))

    # Read class lists and friendly names
    wnids, wnid2name = read_wnids_words(data_root)
    classnames = [wnid2name[w] for w in wnids]

    print(f"Loaded {len(classnames)} classes from Tiny ImageNet.")
    templates = TEMPLATES
    if args.templates < len(TEMPLATES):
        # Truncate templates list if requested (no globals)
        templates = TEMPLATES[: args.templates]
        print(f"Using {len(templates)} prompt templates per class.")

    results = []

    for model_name, pretrained in model_specs:
        print("\n===============================================")
        print(f"Model: {model_name} | Pretrained: {pretrained}")
        print("===============================================")

        # Create model + preprocess
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=args.device)
        tokenizer = open_clip.get_tokenizer(model_name)
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()

        # Dataset / DataLoader
        val_ds = TinyImageNetVal(data_root, preprocess, wnids)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # Build text features (can be done in full precision on CPU/GPU)
        print("Building text features...")
        text_features = build_text_features(model, tokenizer, device, classnames, templates)

        # Evaluate
        print("Evaluating...")
        if args.amp and device.type == "cuda":
            # AMP is most relevant for image encoding; text is tiny
            scaler = torch.cuda.amp.autocast
        else:
            # no-op context manager
            from contextlib import nullcontext
            scaler = nullcontext

        top1, top5 = evaluate(model, val_loader, text_features, device)
        print(f"Zero-shot Top-1: {top1 * 100:.2f}% | Top-5: {top5 * 100:.2f}% | N={len(val_ds)}")
        results.append(((model_name, pretrained), top1, top5))

    print("\nSummary:")
    for (m, pt), t1, t5 in results:
        print(f"  {m:<9} [{pt:<20}]  Top-1 {t1*100:6.2f}% | Top-5 {t5*100:6.2f}%")


if __name__ == "__main__":
    main()
