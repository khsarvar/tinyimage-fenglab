import argparse
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

import open_clip

import torch.nn as nn
from contextlib import nullcontext

# ----------------------------
# Utils
# ----------------------------

def set_all_requires_grad(obj, requires_grad: bool):
    """Works for nn.Module, nn.Parameter, iterables, or None."""
    if obj is None:
        return
    if isinstance(obj, nn.Parameter):
        obj.requires_grad = requires_grad
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            set_all_requires_grad(x, requires_grad)
        return
    # nn.Module path (or any object exposing .parameters())
    params = getattr(obj, "parameters", None)
    if callable(params):
        for p in obj.parameters():
            p.requires_grad = requires_grad
        return
    # fallback: try common attributes
    for name in ("weight", "bias"):
        p = getattr(obj, name, None)
        if isinstance(p, nn.Parameter):
            p.requires_grad = requires_grad


def get_visual_blocks(visual) -> list:
    """
    open_clip has slightly different internals depending on model variant.
    We try the common layouts to retrieve the sequence of transformer blocks.
    """
    # open_clip VisionTransformer often: visual.transformer.resblocks (ModuleList)
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
        return list(blocks)

    # timm-style ViT occasionally exposes .blocks
    if hasattr(visual, "blocks"):
        return list(visual.blocks)

    # fallback: look for something that looks like a list of ResidualAttentionBlock
    candidates = []
    for name, mod in visual.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 and mod[0].__class__.__name__.lower().find("residual") >= 0:
            candidates = list(mod)
            break
    if candidates:
        return candidates

    raise RuntimeError("Could not locate visual transformer blocks on this model.")


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # x: [B, in_dim] normalized CLIP image embedding
        return self.fc(x)


# ----------------------------
# Training / Eval
# ----------------------------
def train_one_epoch(model, head, loader, optimizer, scaler, device, epoch, accumulation_steps=1):
    model.train()
    head.train()

    loss_fn = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # encode_image returns L2-normalized projected features
            feats = model.encode_image(images)            # [B, embed_dim]
            logits = head(feats)                          # [B, num_classes]
            loss = loss_fn(logits, targets) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulation_steps
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, head, loader, device):
    model.eval()
    head.eval()

    loss_fn = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        feats = model.encode_image(images)
        logits = head(feats)
        loss = loss_fn(logits, targets)

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    acc = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to tiny-imagenet-200/ directory")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--unfreeze_k", type=int, default=2, help="Number of last visual transformer blocks to unfreeze")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="open_clip pretrained tag")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1) Load CLIP ViT-B/32
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")  # not used, but handy if extending
    model.to(device)

    # 2) Freeze everything by default
    set_all_requires_grad(model, False)

    # 3) Unfreeze last K visual transformer blocks + final norms/proj
    visual = model.visual
    blocks = get_visual_blocks(visual)
    k = max(0, min(args.unfreeze_k, len(blocks)))

    if k > 0:
        for block in blocks[-k:]:
            set_all_requires_grad(block, True)
        print(f"Unfroze last {k} visual transformer blocks out of {len(blocks)}.")
    else:
        print("Keeping all visual transformer blocks frozen.")

    # Always allow these to learn a bit:
    ln_post = getattr(visual, "ln_post", None)
    proj    = getattr(visual, "proj", None)

    # ln_post is a Module (LayerNorm) on ViT
    set_all_requires_grad(ln_post, True)

    # proj can be a Parameter or a Module depending on variant
    if isinstance(proj, nn.Parameter):
       proj.requires_grad = True
    else:
       set_all_requires_grad(proj, True)

    # (Optional) You can also unfreeze final LayerNorms within the last block(s):
    # for b in blocks[-k:]:
    #     for name, mod in b.named_modules():
    #         if isinstance(mod, nn.LayerNorm):
    #             set_all_requires_grad(mod, True)

    # 4) TinyImageNet datasets (ImageFolder layout)
    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=preprocess)
    val_ds   = datasets.ImageFolder(val_dir,   transform=preprocess)
    num_classes = len(train_ds.classes)
    assert num_classes == 200, f"Expected 200 classes, got {num_classes}"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # 5) Classification head on top of CLIP image embeddings
    embed_dim = model.text_projection.shape[1] if hasattr(model, "text_projection") else model.visual.proj.shape[1]
    head = LinearHead(embed_dim, num_classes).to(device)

    # 6) Optimizer / (optional) scheduler
    # Only optimize unfrozen CLIP params + head
    def trainable_params(m):
        return [p for p in m.parameters() if p.requires_grad]

    param_groups = [
        {"params": trainable_params(model), "lr": args.lr, "weight_decay": args.wd},
        {"params": head.parameters(), "lr": args.lr, "weight_decay": args.wd},
    ]
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and device == "cuda") else None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, head, train_loader, optimizer, scaler, device, epoch, args.accum_steps)
        val_loss, val_acc = evaluate(model, head, val_loader, device)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "head_state": head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
                "classes": train_ds.classes,
            }
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(ckpt, f"checkpoints/openclip_vitb32_tinyimagenet_k{k}_best.pt")
            print(f"Saved new best checkpoint (acc={best_acc:.2f}%).")

    print(f"Best Val Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
