#!/usr/bin/env python3
import os
import argparse
import time
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as tvm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tiny_imagenet_val import TinyImageNetVal


# ------------------- Model -------------------
class TinyNet(nn.Module):
    """
    depth = number of Conv blocks. Each block: Conv3x3 -> BN -> ReLU -> MaxPool2d(2)
    channels progression: [64, 128, 256, 512][:depth]
    """
    def __init__(self, num_classes=200, depth=3):
        super().__init__()
        assert depth in (2, 3, 4), "depth must be one of {2,3,4}"
        chs = [64, 128, 256, 512][:depth]
        layers, in_c = [], 3
        for out_c in chs:
            layers += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_c, num_classes)
        self.depth = depth  # used by resume checks
        self.arch = "tinynet"

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class ResNetTinyINet(nn.Module):
    def __init__(self, variant: str = "resnet18", num_classes: int = 200):
        super().__init__()
        if variant == "resnet18":
            base = tvm.resnet18(weights=None)
        elif variant == "resnet34":
            base = tvm.resnet34(weights=None)
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")

        # Adapt for 64x64
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, num_classes)

        self.base = base
        self.arch = variant  # keeps resume checks working

    def forward(self, x):
        return self.base(x)


def build_model(args):
    if args.arch == "tinynet":
        return TinyNet(num_classes=200, depth=args.depth)
    else:
        return ResNetTinyINet(args.arch, num_classes=200)


# ------------------- Utilities -------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transforms():
    train_tfms = T.Compose([
        T.RandomResizedCrop(64, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tfms = T.Compose([
        T.Resize(72),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tfms, val_tfms


def load_wnid_mapping(data_root):
    """Load wnid -> human-readable label mapping from words.txt (Tiny-ImageNet)."""
    words_path = os.path.join(data_root, "words.txt")
    mapping = {}
    if os.path.isfile(words_path):
        with open(words_path, "r") as f:
            for line in f:
                wnid, desc = line.strip().split("\t")
                mapping[wnid] = desc
    else:
        print(f"[WARN] words.txt not found at {words_path}. Will print raw WNIDs.")
    return mapping


def build_dataloaders(args, device):
    train_dir = os.path.join(args.data_root, "train")
    val_dir   = os.path.join(args.data_root, "val")

    train_tfms, val_tfms = make_transforms()

    train_ds = ImageFolder(root=train_dir, transform=train_tfms)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    val_ds = TinyImageNetVal(val_dir=val_dir, class_to_idx=class_to_idx, transform=val_tfms)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=(args.workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=(args.workers > 0))
    return train_loader, val_loader, idx_to_class


def topk_accuracy(logits, targets, ks=(1, 5)):
    with torch.no_grad():
        maxk = max(ks)
        B = targets.size(0)
        _, pred = logits.topk(maxk, dim=1)  # (B, maxk)
        pred = pred.t()                     # (maxk, B)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = {}
        for k in ks:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res[k] = (correct_k / B).item()
        return res


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=True)  # full checkpoint
    else:
        model.load_state_dict(state, strict=True)           # weights-only
    print(f"Loaded weights from: {ckpt_path}")


def load_training_state(ckpt_path, model, optimizer, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    # Architecture must match if present
    if "arch" in saved_args:
        assert getattr(model, "arch", None) == saved_args["arch"], \
            f"Resume arch mismatch: ckpt arch={saved_args['arch']} vs model arch={getattr(model,'arch',None)}"

    # Only enforce depth for TinyNet
    if saved_args.get("arch", "tinynet") == "tinynet" and "depth" in saved_args:
        assert getattr(model, "depth", None) == saved_args["depth"], \
            f"Resume depth mismatch: ckpt depth={saved_args['depth']} vs model depth={getattr(model,'depth',None)}"

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_top1   = ckpt.get("best_top1", 0.0)
    return start_epoch, best_top1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running, t1_sum, t5_sum, n = 0.0, 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        accs = topk_accuracy(logits, labels, ks=(1, 5))
        bsz = imgs.size(0)
        running += loss.item() * bsz
        t1_sum += accs[1] * bsz
        t5_sum += accs[5] * bsz
        n += bsz
    return running / n, t1_sum / n, t5_sum / n


def predict_single(model, img_path, device, class_names, wnid_to_words):
    _, val_tfms = make_transforms()
    img = Image.open(img_path).convert("RGB")
    x = val_tfms(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)[0]
        top5 = torch.topk(probs, k=5)
    out = []
    for prob, idx in zip(top5.values, top5.indices):
        wnid = class_names[idx.item()]
        label = wnid_to_words.get(wnid, wnid)
        out.append((label, float(prob)))
    return out


# ------------------- Run ID & Paths -------------------
def model_name(args):
    if args.arch == "tinynet":
        return f"TinyNet_d{args.depth}"
    elif args.arch == "resnet18":
        return "ResNet18"
    elif args.arch == "resnet34":
        return "ResNet34"
    else:
        return args.arch

def make_run_id(args):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{model_name(args)}_bs{args.batch_size}_lr{args.lr:g}"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def derive_paths(args, run_id):
    """
    Returns:
      plots_dir_run, ckpt_dir_run, metrics_dir_run, weights_path, metrics_path
    """
    plots_dir_run   = os.path.join(args.plots_outdir, run_id)
    ckpt_dir_run    = os.path.join(args.ckpt_dir, run_id)
    metrics_dir_run = os.path.join(args.metrics_dir, run_id)

    ensure_dir(plots_dir_run)
    ensure_dir(ckpt_dir_run)
    ensure_dir(metrics_dir_run)

    # If user passed explicit filenames, honor them; else derive.
    weights_path = args.out if args.out else os.path.join(ckpt_dir_run,   f"{run_id}_best.pth")
    metrics_path = args.metrics if args.metrics else os.path.join(metrics_dir_run, f"{run_id}_metrics.json")

    return plots_dir_run, ckpt_dir_run, metrics_dir_run, weights_path, metrics_path


# ------------------- Train / Eval loops -------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running, t1_sum, t5_sum, n = 0.0, 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        accs = topk_accuracy(logits, labels, ks=(1, 5))
        bsz = imgs.size(0)
        running += loss.item() * bsz
        t1_sum += accs[1] * bsz
        t5_sum += accs[5] * bsz
        n += bsz

    return running / n, t1_sum / n, t5_sum / n


# ------------------- History / Plotting -------------------
def load_history_json(path):
    with open(path, "r") as f:
        return json.load(f)

def slice_history_upto_epoch(H, last_epoch):
    """Keep entries with epoch <= last_epoch (monotonic epochs)."""
    def _slice(arr):
        # assume same length as H["epoch"]
        k = 0
        for i, e in enumerate(H["epoch"]):
            if e <= last_epoch: k = i + 1
        return arr[:k]
    return {
        k: (_slice(v) if k != "epoch" else [e for e in H["epoch"] if e <= last_epoch])
        for k, v in H.items()
    }

def merge_histories(prev, new):
    """Concatenate per-key lists; assumes no overlapping epochs in 'new'."""
    out = {k: (prev.get(k, []) + new.get(k, [])) for k in set(prev) | set(new)}
    return out

def init_history():
    return {
        "epoch": [],
        "train_loss": [], "val_loss": [],
        "train_top1": [], "val_top1": [],
        "train_top5": [], "val_top5": [],
        "epoch_time": []
    }

def update_history(H, epoch, tr_loss, va_loss, tr_t1, va_t1, tr_t5, va_t5, dt):
    H["epoch"].append(epoch)
    H["train_loss"].append(tr_loss);   H["val_loss"].append(va_loss)
    H["train_top1"].append(tr_t1);     H["val_top1"].append(va_t1)
    H["train_top5"].append(tr_t5);     H["val_top5"].append(va_t5)
    H["epoch_time"].append(dt)

def save_history_json(H, path):
    with open(path, "w") as f:
        json.dump(H, f, indent=2)

def _stamp_plot(fig, run_id):
    fig.text(0.99, 0.01, run_id, ha="right", va="bottom", fontsize=8, alpha=0.7)

def plot_curves(H, outdir, run_id):
    ensure_dir(outdir)

    # Loss
    fig = plt.figure()
    plt.plot(H["epoch"], H["train_loss"], label="Train Loss")
    plt.plot(H["epoch"], H["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Loss vs Epochs · {run_id}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    _stamp_plot(fig, run_id)
    loss_path = os.path.join(outdir, f"{run_id}_loss_curve.png")
    plt.savefig(loss_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Accuracy (Top-1)
    fig = plt.figure()
    plt.plot(H["epoch"], [x*100 for x in H["train_top1"]], label="Train Top-1")
    plt.plot(H["epoch"], [x*100 for x in H["val_top1"]],   label="Val Top-1")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title(f"Top-1 Accuracy vs Epochs · {run_id}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    _stamp_plot(fig, run_id)
    acc1_path = os.path.join(outdir, f"{run_id}_acc_curve_top1.png")
    plt.savefig(acc1_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"[PLOTS] Saved: {loss_path}")
    print(f"[PLOTS] Saved: {acc1_path}")


# ------------------- Main -------------------
def main(args):
    # Gets device information for ...
    device = get_device()
    print("Using device:", device)

    # Creates an id for the run, so later could be used for saving the results
    run_id = make_run_id(args)
    plots_dir, ckpt_dir, metrics_dir, weights_path, metrics_path = derive_paths(args, run_id)
    print(f"[RUN] {run_id}")
    print(f"[IO ] checkpoints dir: {ckpt_dir}")
    print(f"[IO ] metrics dir    : {metrics_dir}")
    print(f"[IO ] plots dir      : {plots_dir}")
    print(f"[IO ] checkpoint file: {weights_path}")
    print(f"[IO ] metrics file   : {metrics_path}")

    train_loader, val_loader, idx_to_class = build_dataloaders(args, device)
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    wnid_to_words = load_wnid_mapping(args.data_root)

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Evaluate-only path ----
    if args.eval_only:
        if not args.weights:
            raise ValueError("--eval-only requires --weights <path>")
        load_checkpoint(model, args.weights, device)
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device)
        print(f"[EVAL] val_loss={val_loss:.4f} | top1={top1*100:.2f}% | top5={top5*100:.2f}%")
        if args.predict:
            if not os.path.isfile(args.predict):
                raise FileNotFoundError(f"No such image: {args.predict}")
            res = predict_single(model, args.predict, device, class_names, wnid_to_words)
            print(f"Top-5 prediction for: {args.predict}")
            for name, p in res:
                print(f"  {name}: {p*100:.2f}%")
        return

    # ---- Resume training (optional) ----
    start_epoch, best_top1 = 1, 0.0
    if args.resume:
        start_epoch, best_top1 = load_training_state(args.resume, model, optimizer, device)

    # ---- History bootstrapping ----
    # Always create a *new* run_id & output paths (so old files remain)
    history = init_history()

    prior_history = None
    if args.resume_metrics:
        if os.path.isfile(args.resume_metrics):
            prior_history = load_history_json(args.resume_metrics)
        else:
            print(f"[WARN] --resume-metrics not found: {args.resume_metrics}")

    # If we’re resuming from epoch K, only keep prior entries up to K-1,
    # then we’ll append new epochs K..args.epochs
    if prior_history is not None:
        keep_upto = (start_epoch - 1)
        prior_trimmed = slice_history_upto_epoch(prior_history, keep_upto)
        # Seed current history with prior data
        history = prior_trimmed

    # ---- Train ----
    print("Training started..")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_top1, va_top5 = evaluate(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        if va_top1 > best_top1:
            best_top1 = va_top1
            torch.save({
               "model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "epoch": epoch,
               "best_top1": best_top1,
               "args": vars(args),  # record for safe resume
            }, weights_path)

        update_history(history, epoch, tr_loss, va_loss, tr_top1, va_top1, tr_top5, va_top5, dt)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} | train_top1={tr_top1*100:.2f}% | train_top5={tr_top5*100:.2f}% | "
            f"val_loss={va_loss:.4f} | val_top1={va_top1*100:.2f}% | val_top5={va_top5*100:.2f}% | "
            f"epoch_time={dt:.1f}s"
        )

    print("Best val top1:", f"{best_top1*100:.2f}%")
    print("Saved best weights to:", weights_path)

    save_history_json(history, metrics_path)
    print(f"[METRICS] Wrote {metrics_path}")
    if not args.no_plots:
        plot_curves(history, plots_dir, run_id)

    # Optional single-image prediction after training (uses best saved weights)
    if args.predict:
        if not os.path.isfile(args.predict):
            raise FileNotFoundError(f"No such image: {args.predict}")
        load_checkpoint(model, weights_path, device)
        res = predict_single(model, args.predict, device, class_names, wnid_to_words)
        print(f"Top-5 prediction for: {args.predict}")
        for name, p in res:
            print(f"  {name}: {p*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="tiny-imagenet-200",
                        help="Path to tiny-imagenet-200 directory")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")

    # Architecture / Depth
    parser.add_argument("--arch", type=str, choices=["tinynet", "resnet18", "resnet34"],
                        default="tinynet", help="Backbone architecture.")
    parser.add_argument("--depth", type=int, choices=[2,3,4], default=3,
                        help="Number of Conv blocks for TinyNet (ignored for ResNets).")

    # Resume / eval / inference
    parser.add_argument("--resume-metrics", type=str, default="", help="Path to a previous metrics.json to carry forward when resuming")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the given weights")
    parser.add_argument("--weights", type=str, default="", help="Weights to evaluate (with --eval-only)")
    parser.add_argument("--predict", type=str, default="", help="Path to a single image for prediction")

    # Output dirs (each gets a per-run subfolder)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--metrics-dir", type=str, default="metrics", help="Base directory for metrics JSON")
    parser.add_argument("--plots-outdir", type=str, default="plots", help="Base directory for plots")

    # Optional explicit filenames (bypass auto naming if provided)
    parser.add_argument("--out", type=str, default="", help="Explicit checkpoint filepath (optional)")
    parser.add_argument("--metrics", type=str, default="", help="Explicit metrics JSON filepath (optional)")

    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")

    args = parser.parse_args()
    main(args)
