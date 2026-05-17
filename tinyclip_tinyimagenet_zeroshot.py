#!/usr/bin/env python
"""
Zero-shot TinyImageNet classification using TinyCLIP from Hugging Face.

Default checkpoint:
  wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M

Example:
  python tinyclip_tinyimagenet_zeroshot.py --data-root tiny-imagenet-200 --device cuda
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import open_clip


DEFAULT_HF_MODEL_ID = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"

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


def configure_ssl_cert_env() -> None:
    # Remove stale SSL paths injected by cluster environment modules.
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")
    if ssl_cert_file and not Path(ssl_cert_file).is_file():
        os.environ.pop("SSL_CERT_FILE", None)

    ssl_cert_dir = os.environ.get("SSL_CERT_DIR")
    if ssl_cert_dir and not Path(ssl_cert_dir).is_dir():
        os.environ.pop("SSL_CERT_DIR", None)

    # Prefer certifi CA bundle when available.
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi

            os.environ["SSL_CERT_FILE"] = certifi.where()
        except Exception:
            pass


def read_wnids_words(data_root: Path) -> Tuple[List[str], Dict[str, str]]:
    wnids_path = data_root / "wnids.txt"
    words_path = data_root / "words.txt"
    if not wnids_path.exists() or not words_path.exists():
        raise FileNotFoundError("Missing wnids.txt or words.txt under TinyImageNet root.")

    wnids = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
    mapping: Dict[str, str] = {}
    for line in words_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        wnid, names = parts
        mapping[wnid] = names.split(",")[0].strip()
    for wnid in wnids:
        mapping.setdefault(wnid, wnid)
    return wnids, mapping


class TinyImageNetVal(Dataset):
    def __init__(self, data_root: Path, preprocess, wnids: List[str]):
        self.preprocess = preprocess
        img_dir = data_root / "val" / "images"
        ann_path = data_root / "val" / "val_annotations.txt"
        if not ann_path.exists():
            raise FileNotFoundError("Missing val/val_annotations.txt.")

        wnid_to_idx = {w: i for i, w in enumerate(wnids)}
        self.samples: List[Tuple[Path, int]] = []
        for line in ann_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(" ")
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            if wnid not in wnid_to_idx:
                continue
            self.samples.append((img_dir / img_name, wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return self.preprocess(img), target


def unpack_model_preprocess(ret):
    if not isinstance(ret, tuple):
        raise RuntimeError("Unexpected return type from OpenCLIP model constructor.")
    if len(ret) == 2:
        model, preprocess = ret
    elif len(ret) == 3:
        model, _, preprocess = ret
    else:
        raise RuntimeError(f"Unexpected tuple length from OpenCLIP API: {len(ret)}")
    return model, preprocess


class HFCLIPAdapter(torch.nn.Module):
    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model
        self.logit_scale = hf_model.logit_scale

    def to(self, *args, **kwargs):
        self.hf_model.to(*args, **kwargs)
        return self

    def eval(self):
        self.hf_model.eval()
        return self

    @staticmethod
    def _as_feature_tensor(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if hasattr(output, "text_embeds") and torch.is_tensor(output.text_embeds):
            return output.text_embeds
        if hasattr(output, "image_embeds") and torch.is_tensor(output.image_embeds):
            return output.image_embeds
        if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
            return output.pooler_output
        if isinstance(output, Mapping):
            for key in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if torch.is_tensor(value):
                    return value
        raise TypeError(f"Expected tensor-like model output, got {type(output).__name__}")

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        output = self.hf_model.get_image_features(pixel_values=image_tensor)
        return self._as_feature_tensor(output)

    def encode_text(self, token_batch) -> torch.Tensor:
        if isinstance(token_batch, dict):
            token_inputs = token_batch
        elif isinstance(token_batch, Mapping):
            token_inputs = dict(token_batch)
        elif hasattr(token_batch, "items"):
            token_inputs = dict(token_batch.items())
        else:
            raise TypeError("HFCLIPAdapter expected tokenizer output as a mapping of tensors.")
        output = self.hf_model.get_text_features(**token_inputs)
        return self._as_feature_tensor(output)


def load_tinyclip_from_transformers(model_source: str, device: str):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:
        raise RuntimeError(
            "OpenCLIP HF loading failed and transformers fallback is unavailable. "
            "Install transformers in this environment to load this HF CLIP repo."
        ) from exc

    hf_model = CLIPModel.from_pretrained(model_source).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_source)

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    def tokenizer(prompts: List[str]):
        return processor(text=prompts, return_tensors="pt", padding=True, truncation=True)

    return HFCLIPAdapter(hf_model), tokenizer, preprocess


def load_tinyclip(hf_model_id: str, device: str, local_model_dir: str | None = None):
    if local_model_dir:
        local_path = Path(local_model_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local model directory not found: {local_path}")
        return load_tinyclip_from_transformers(str(local_path), device)

    model_ref = f"hf-hub:{hf_model_id}"
    try:
        if hasattr(open_clip, "create_model_from_pretrained"):
            model, preprocess = unpack_model_preprocess(
                open_clip.create_model_from_pretrained(model_ref, device=device)
            )
        else:
            model, preprocess = unpack_model_preprocess(
                open_clip.create_model_and_transforms(model_ref, pretrained=None, device=device)
            )
        try:
            tokenizer = open_clip.get_tokenizer(model_ref)
        except Exception:
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model, tokenizer, preprocess
    except Exception as exc:
        msg = str(exc)
        if "open_clip_config.json" not in msg and "HF Hub" not in msg:
            raise
        print("OpenCLIP HF format not found; falling back to transformers CLIP loader.")
        return load_tinyclip_from_transformers(hf_model_id, device)


def warmup_gpu(model, device, n: int = 3) -> None:
    """Run dummy image forward passes to warm up CUDA kernels before timing."""
    if device.type != "cuda":
        return
    dummy = torch.randn(4, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(n):
            model.encode_image(dummy)
    torch.cuda.synchronize()


@torch.no_grad()
def build_text_features(model, tokenizer, classnames: List[str], templates: List[str], device: torch.device):
    class_features = []
    for class_name in classnames:
        prompts = [tpl.format(class_name) for tpl in templates]
        tokens = tokenizer(prompts)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        elif isinstance(tokens, dict):
            tokens = {k: v.to(device) for k, v in tokens.items()}
        else:
            raise TypeError("Tokenizer output must be Tensor-like or a dict of tensors.")
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        pooled = F.normalize(text_features.mean(dim=0), dim=-1)
        class_features.append(pooled)
    return torch.stack(class_features, dim=0)


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, text_features: torch.Tensor, device: torch.device):
    model.eval()
    text_features = text_features.to(device)
    logit_scale = model.logit_scale.exp()
    total, top1, top5 = 0, 0, 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        image_features = F.normalize(model.encode_image(images), dim=-1)
        logits = logit_scale * image_features @ text_features.t()
        _, pred = logits.topk(5, dim=-1)
        top1 += (pred[:, 0] == targets).sum().item()
        top5 += (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    return top1 / total, top5 / total


def parse_args():
    parser = argparse.ArgumentParser(description="TinyCLIP zero-shot TinyImageNet")
    parser.add_argument("--data-root", type=str, default="tiny-imagenet-200")
    parser.add_argument("--hf-model-id", type=str, default=DEFAULT_HF_MODEL_ID)
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default=None,
        help="Local directory containing a CLIP model saved from Hugging Face. "
        "If set, no network download is attempted.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--templates", type=int, default=len(TEMPLATES))
    return parser.parse_args()


def main():
    configure_ssl_cert_env()
    args = parse_args()
    data_root = Path(args.data_root)
    device = torch.device(args.device)

    wnids, wnid_to_name = read_wnids_words(data_root)
    classnames = [wnid_to_name[w] for w in wnids]
    templates = TEMPLATES[: max(1, min(args.templates, len(TEMPLATES)))]

    if args.local_model_dir:
        print(f"Loading TinyCLIP from local dir: {args.local_model_dir}")
    else:
        print(f"Loading TinyCLIP from HF: {args.hf_model_id}")
    t0 = time.time()
    model, tokenizer, preprocess = load_tinyclip(
        args.hf_model_id, args.device, local_model_dir=args.local_model_dir
    )
    model = model.to(device).eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # Warm up CUDA kernels so timing is not distorted by JIT / lazy init
    warmup_gpu(model, device)

    dataset = TinyImageNetVal(data_root, preprocess, wnids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Building text features for {len(classnames)} classes using {len(templates)} prompts/class...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    text_features = build_text_features(model, tokenizer, classnames, templates, device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    text_time = time.time() - t0
    print(f"Text features built in {text_time:.2f}s")

    print("Running evaluation...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    top1, top5 = evaluate(model, loader, text_features, device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    eval_time = time.time() - t0
    total_time = load_time + text_time + eval_time
    print(f"Evaluation completed in {eval_time:.2f}s")
    print(f"Zero-shot TinyImageNet | Top-1: {top1 * 100:.2f}% | Top-5: {top5 * 100:.2f}% | N={len(dataset)}")
    print(f"Timing: load={load_time:.2f}s | text={text_time:.2f}s | eval={eval_time:.2f}s | total={total_time:.2f}s")


if __name__ == "__main__":
    main()
