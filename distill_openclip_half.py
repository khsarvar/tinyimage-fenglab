#!/usr/bin/env python3
import argparse
import math
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image

import open_clip
from open_clip import factory
from contextlib import nullcontext


# ---------------------------
# Prompts
# ---------------------------

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


# ---------------------------
# TinyImageNet helpers
# ---------------------------

def read_wnids_words(data_root: Path) -> Tuple[List[str], Dict[str, str]]:
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
            primary = names.split(",")[0].strip()
            wmap[wnid] = primary
        except ValueError:
            continue
    for w in wnids:
        wmap.setdefault(w, w)
    return wnids, wmap


class TinyImageNetValFromAnnotations(Dataset):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
        return self.preprocess(img), y


def remap_imagefolder_to_canonical(ds: datasets.ImageFolder, canonical_classes: List[str]):
    canonical = {w: i for i, w in enumerate(canonical_classes)}
    new_samples = []
    for path, _y in ds.samples:
        p = Path(path)
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


# ---------------------------
# Distillation helpers
# ---------------------------

def get_autocast_and_scaler(device_type: str, use_amp: bool):
    if not use_amp:
        return nullcontext(), None
    if device_type == "cuda":
        return torch.cuda.amp.autocast(), torch.cuda.amp.GradScaler()
    if device_type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16), None
    return nullcontext(), None


def make_half_config(model_cfg: Dict) -> Dict:
    cfg = deepcopy(model_cfg)
    cfg["embed_dim"] = max(1, int(cfg["embed_dim"] // 2))

    vcfg = cfg["vision_cfg"]
    tcfg = cfg["text_cfg"]

    if isinstance(vcfg.get("layers"), (tuple, list)):
        vcfg["layers"] = tuple(max(1, int(x // 2)) for x in vcfg["layers"])
    else:
        vcfg["layers"] = max(1, int(vcfg["layers"] // 2))
    vcfg["width"] = max(1, int(vcfg["width"] // 2))

    tcfg["layers"] = max(1, int(tcfg["layers"] // 2))
    tcfg["width"] = max(1, int(tcfg["width"] // 2))
    tcfg["heads"] = max(1, int(tcfg["heads"] // 2))

    # Ensure width is divisible by heads for the text transformer.
    if tcfg["width"] % tcfg["heads"] != 0:
        for h in range(tcfg["heads"], 0, -1):
            if tcfg["width"] % h == 0:
                tcfg["heads"] = h
                break

    return cfg


def distill_kl(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    bsz = logits.size(0)
    targets = torch.arange(bsz, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def build_text_batch(labels: torch.Tensor, classnames: List[str], templates: List[str], mode: str) -> List[str]:
    idxs = labels.detach().cpu().tolist()
    if mode == "single":
        tmpl = templates[0]
        return [tmpl.format(classnames[i]) for i in idxs]
    return [random.choice(templates).format(classnames[i]) for i in idxs]


@torch.no_grad()
def build_text_features(model, tokenizer, classnames: List[str], templates: List[str], device: torch.device):
    model.eval()
    text_embeds = []
    for cname in classnames:
        prompts = [t.format(cname) for t in templates]
        tokens = tokenizer(prompts).to(device)
        class_embeds = model.encode_text(tokens)
        class_embeds = F.normalize(class_embeds, dim=-1)
        class_embed = class_embeds.mean(dim=0)
        class_embed = F.normalize(class_embed, dim=-1)
        text_embeds.append(class_embed)
    return torch.stack(text_embeds, dim=0)


@torch.no_grad()
def evaluate_zero_shot(model, dataloader: DataLoader, text_features: torch.Tensor, device: torch.device):
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
        _, pred = logits.topk(5, dim=-1)
        top1 += (pred[:, 0] == targets).sum().item()
        in_top5 = (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        top5 += in_top5
        total += targets.size(0)

    return top1 / total, top5 / total


def make_run_id(args) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = args.teacher_model.replace("/", "-")
    return f"{ts}_{model_slug}_half_clip_distill"


def parse_args():
    ap = argparse.ArgumentParser(description="Distill CLIP with half width/layers using OpenCLIP.")
    ap.add_argument("--data_root", type=str, default="tiny-imagenet-200")
    ap.add_argument("--teacher_model", type=str, default="ViT-B-32")
    ap.add_argument("--teacher_pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--distill_weight", type=float, default=1.0)
    ap.add_argument("--clip_weight", type=float, default=1.0)
    ap.add_argument("--embed_weight", type=float, default=0.0)
    ap.add_argument("--template_mode", type=str, choices=["single", "random"], default="random")
    ap.add_argument("--val_from_annotations", action="store_true")
    ap.add_argument("--eval_each", type=int, default=1, help="Evaluate every N epochs; 0 disables eval")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    run_id = args.run_name.strip() or make_run_id(args)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_id}_student_half.pt"

    # Teacher (full) model + preprocess
    teacher, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        args.teacher_model, pretrained=args.teacher_pretrained
    )
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student (half width/half layers) model
    base_cfg = factory.get_model_config(args.teacher_model)
    student_cfg = make_half_config(base_cfg)
    student = open_clip.create_model(
        args.teacher_model,
        pretrained=None,
        load_weights=False,
        **student_cfg,
    )
    student.to(device)

    tokenizer = open_clip.get_tokenizer(args.teacher_model)

    # Dataset / loaders
    data_root = Path(args.data_root)
    wnids, wnid2name = read_wnids_words(data_root)
    classnames = [wnid2name[w] for w in wnids]

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_preprocess)
    missing = set(wnids) - set(train_ds.classes)
    if missing:
        raise RuntimeError(f"Train set missing classes: {sorted(list(missing))[:5]} ...")
    remap_imagefolder_to_canonical(train_ds, wnids)

    if args.eval_each:
        if args.val_from_annotations:
            val_ds = TinyImageNetValFromAnnotations(data_root, val_preprocess, wnids)
        else:
            val_ds = datasets.ImageFolder(data_root / "val", transform=val_preprocess)
            if len(val_ds.classes) == 200:
                remap_imagefolder_to_canonical(val_ds, wnids)
            else:
                raise RuntimeError(
                    "val/ does not look like an ImageFolder with 200 class dirs. "
                    "Use --val_from_annotations to read the original layout."
                )
    else:
        val_ds = None

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    ) if val_ds is not None else None

    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    autocast_ctx, scaler = get_autocast_and_scaler(device, args.fp16)

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            texts = build_text_batch(labels, classnames, TEMPLATES, args.template_mode)
            tokens = tokenizer(texts).to(device)

            with torch.no_grad():
                with autocast_ctx:
                    t_img = teacher.encode_image(images)
                    t_txt = teacher.encode_text(tokens)
                    t_img = F.normalize(t_img, dim=-1)
                    t_txt = F.normalize(t_txt, dim=-1)
                    t_logits = teacher.logit_scale.exp() * (t_img @ t_txt.t())

            with autocast_ctx:
                s_img = student.encode_image(images)
                s_txt = student.encode_text(tokens)
                s_img = F.normalize(s_img, dim=-1)
                s_txt = F.normalize(s_txt, dim=-1)
                s_logits = student.logit_scale.exp() * (s_img @ s_txt.t())

                loss_clip = clip_loss(s_logits)
                loss_kd = 0.5 * (
                    distill_kl(s_logits, t_logits, args.temperature) +
                    distill_kl(s_logits.t(), t_logits.t(), args.temperature)
                )
                if args.embed_weight > 0:
                    if s_img.shape[-1] != t_img.shape[-1] or s_txt.shape[-1] != t_txt.shape[-1]:
                        raise RuntimeError(
                            "embed_weight > 0 but student/teacher embedding dims differ. "
                            "Set --embed_weight 0 or add a projection layer."
                        )
                    loss_embed = 0.5 * (F.mse_loss(s_img, t_img) + F.mse_loss(s_txt, t_txt))
                else:
                    loss_embed = torch.zeros((), device=images.device)
                raw_loss = (
                    args.clip_weight * loss_clip +
                    args.distill_weight * loss_kd +
                    args.embed_weight * loss_embed
                )

            loss = raw_loss / max(1, args.accum_steps)
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1
            if accum_counter >= args.accum_steps:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    student.logit_scale.clamp_(0, math.log(100))
                accum_counter = 0

            epoch_loss += raw_loss.item() * images.size(0)
            step_count += images.size(0)

        if accum_counter > 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                student.logit_scale.clamp_(0, math.log(100))

        epoch_loss = epoch_loss / max(1, step_count)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} - loss: {epoch_loss:.4f} - time: {dt:.1f}s")

        if val_loader is not None and args.eval_each and epoch % args.eval_each == 0:
            text_features = build_text_features(student, tokenizer, classnames, TEMPLATES, device)
            top1, top5 = evaluate_zero_shot(student, val_loader, text_features, device)
            print(f"Val zero-shot - top1: {top1:.4f} top5: {top5:.4f}")
            if top1 > best_top1:
                best_top1 = top1
                torch.save(
                    {
                        "epoch": epoch,
                        "model": student.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                        "top1": top1,
                        "top5": top5,
                    },
                    ckpt_path,
                )
                print(f"Saved best student checkpoint to: {ckpt_path}")

    if val_loader is None:
        torch.save(
            {"epoch": args.epochs, "model": student.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)},
            ckpt_path,
        )
        print(f"Saved student checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
