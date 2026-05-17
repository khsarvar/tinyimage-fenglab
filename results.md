# FengLab — Project Log & Results

Zero-shot and supervised classification on [Tiny ImageNet](https://cs231n.stanford.edu/tiny-imagenet-200.zip) (200 classes, 10,000 validation images).
Experiments run on NYU HPC via SLURM, GPU constraint `l40s|h200`.

---

## Project Goal

Build a **2-stage zero-shot classification pipeline** on Tiny ImageNet:

- **Stage 1** — small/fast model (TinyCLIP ViT-61M-32 or ViT-B/32) classifies all samples quickly
- **Stage 2** — larger model (ViT-B/16 or stronger) re-classifies only the samples where stage 1 is not competent

**Objectives:** Maximize Top-1 accuracy while reducing total compute — the expensive model only runs on hard cases.
**Emphasis:** Zero-shot throughout (no fine-tuning as the primary path).
**Open question:** What defines "not competent" for stage 1 — softmax confidence threshold, top-1/top-5 disagreement, or per-class accuracy profiling, variance of top 5, gap: top 1 - top 5. TBD as experiments progress.

---

## Table of Contents

1. [Phase 1 — Supervised Baselines (Sep–Oct 2025)](#phase-1--supervised-baselines-sep-oct-2025)
2. [Phase 2 — OpenCLIP Fine-tuning (Nov 2025)](#phase-2--openclip-fine-tuning-nov-2025)
3. [Phase 3 — Knowledge Distillation (Jan 2026)](#phase-3--knowledge-distillation-jan-2026)
4. [Phase 4 — TinyCLIP Zero-Shot (Feb 2026)](#phase-4--tinyclip-zero-shot-feb-2026)
5. [Phase 5 — Per-phase Timing & Model Comparison (Apr–May 2026)](#phase-5--per-phase-timing--model-comparison-apr-may-2026)
6. [Summary Comparison](#summary-comparison)

---

## Phase 1 — Supervised Baselines (Sep–Oct 2025)

**Goal:** Establish supervised training baselines on Tiny ImageNet before exploring CLIP-based methods.

**Models trained:**
- Custom TinyNet (d3/d4 depth variants) — lightweight CNN
- ResNet-18 and ResNet-34 (torchvision pretrained backbone, fine-tuned)

**Key changes:**
- `2025-09-29` — Initial commit, project scaffold
- `2025-10-03` — Added env setup, `requirements.txt`, `.gitignore`
- `2025-10-07` — Added ResNet-18 and ResNet-34 to `test.py` / `train_tinynet.py`

**Results (supervised, best val Top-1):**

| Model | Batch Size | Best Top-1 | Best Top-5 | Best Epoch |
|---|---|---|---|---|
| TinyNet d3 | 128 | 43.55% | 67.86% | 150 |
| TinyNet d4 | 512 | 45.97% | 71.47% | 48 |
| ResNet-18 | 512 | 54.51% | 76.16% | 268 |
| ResNet-34 | 512 | 53.82% | 76.23% | 90 |

ResNet-18 narrowly outperformed ResNet-34, likely due to ResNet-34 needing more epochs.
TinyNet d4 improved over d3 (+2.4 pp Top-1) at the cost of more parameters.

---

## Phase 2 — OpenCLIP Fine-tuning (Nov 2025)

**Goal:** Fine-tune a pre-trained ViT-B/32 CLIP model on Tiny ImageNet and measure how much supervision improves zero-shot accuracy.

**Key changes:**
- `2025-11-03` — Added `openclip_imagenet.py` (zero-shot eval) and `finetune_OpenClip_imagenet.py`
- `2025-11-08` — Fixed validation data loader to correctly partition val images by label
- `2025-11-14` — Added per-epoch timing and JSON metrics saving; added `plots/` graph generation

**Setup:**
- Model: `ViT-B/32` from OpenCLIP (`laion2b_s34b_b79k`)
- Batch size: 256, `--unfreeze_k` controls how many of the last visual transformer blocks are unfrozen

**Results (fine-tuned ViT-B/32, best val Top-1):**

| Run | Unfreeze k | Best Top-1 | Best Top-5 | Best Epoch |
|---|---|---|---|---|
| 20251114-102025 | k=2 | 79.05% | 93.86% | 3 |
| 20251114-102109 | k=1 | 79.17% | 93.15% | 4 |
| 20251114-114951 | k=3 | **79.52%** | **93.84%** | 3 |
| 20251114-124417 | k=1 | 78.57% | 93.61% | 3 |
| 20251114-124417 | k=2 | 79.08% | 92.30% | 12 |
| 20251114-124417 | k=3 | 79.04% | 93.06% | 4 |

Fine-tuning with k=3 unfrozen blocks converged fastest and reached the highest Top-1 (~79.5%).
All runs converged within a handful of epochs — CLIP's pre-trained representations transfer easily.

---

## Phase 3 — Knowledge Distillation (Jan 2026)

**Goal:** Distill the fine-tuned ViT-B/32 teacher into a "half-capacity" student model to explore compression.

**Key changes:**
- `2026-01-22` — Added `distill_openclip_half.py` and `scripts/run_distill_openclip_half.sbatch`
- `2026-01-22` — Pilot 10-epoch run (jobs `4527238`, `4527250`)
- `2026-01-23` — Full 50-epoch run (job `4539170`), results committed

**Setup:**
- Teacher: ViT-B/32 (`laion2b_s34b_b79k`)
- Student: ViT-B/32 with half the embedding width
- Training: cosine-similarity loss between teacher and student embeddings
- Dataset: Tiny ImageNet train split, ~250s/epoch on GPU

**Distillation run (50 epochs, job `4539170`):**

| Epoch | Loss | Top-1 | Top-5 |
|---|---|---|---|
| 1 | 11.22 | 0.50% | 2.50% |
| 10 | 7.84 | 20.61% | 45.17% |
| 20 | 6.09 | 29.94% | 52.86% |
| 30 | 4.82 | 33.44% | 58.29% |
| 44 | 3.12 | **34.70%** | **58.09%** ← best |
| 50 | 3.01 | 34.53% | 57.85% |

**Observations:**
- Loss decreased steadily (11.2 → 3.0) but accuracy plateaued around epoch 33–44 (~34–35% Top-1).
- This is well below the teacher zero-shot baseline (67.72%) and indicates the half-capacity student struggles to fully absorb the teacher's embedding space.
- Distillation outcome is a known open research challenge when drastically reducing width.

---

## Phase 4 — TinyCLIP Zero-Shot (Feb 2026)

**Goal:** Benchmark `wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M` zero-shot performance against full ViT-B models, as a reference for what a purpose-built compact CLIP achieves vs. our distilled model.

**Key changes:**
- `2026-02-27` — Added `tinyclip_tinyimagenet_zeroshot.py` and `scripts/run_tinyclip_zeroshot.sbatch`
- Initial runs (`2533917`, `2537783`) failed due to stale SSL env vars injected by the cluster modules — fixed by detecting and unsetting `SSL_CERT_FILE` / `SSL_CERT_DIR` if the paths don't exist, then falling back to `certifi`
- Subsequent runs (`2905163`, `2905437`) hit HuggingFace download issues — fixed by caching the model locally
- Job `2905891` — first clean successful run

**Result (job `2905891`):**

| Model | Top-1 | Top-5 |
|---|---|---|
| TinyCLIP ViT-61M-32 | 65.24% | 85.86% |

---

## Phase 5 — Per-phase Timing & Model Comparison (Apr–May 2026)

**Goal:** Add structured load / text / eval timing to both scripts, then run all three models on dedicated jobs for a clean apples-to-apples comparison.

**Key changes:**
- `2026-04-12` — Added per-phase timing (`load`, `text`, `eval`) to `openclip_imagenet.py` and `tinyclip_tinyimagenet_zeroshot.py`
- Added 13-prompt template ensemble to TinyCLIP script (same set used in original CLIP paper)
- Split into separate sbatch scripts per model: `run_vitb32_zeroshot.sbatch`, `run_vitb16_zeroshot.sbatch`, `run_tinyclip_zeroshot.sbatch`

**Latest runs (May 2026):**

| Model | Job ID | Script |
|---|---|---|
| TinyCLIP ViT-61M-32 | 6160695, 6161002 | `run_tinyclip_zeroshot.sbatch` |
| ViT-B/32 | 6160692 | `run_vitb32_zeroshot.sbatch` |
| ViT-B/16 | 6160693 | `run_vitb16_zeroshot.sbatch` |

**Results:**

| Model | Checkpoint | Top-1 | Top-5 | load (s) | text (s) | eval (s) | total (s) |
|---|---|---|---|---|---|---|---|
| TinyCLIP ViT-61M-32 | HF `wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M` | 65.25% | 85.86% | 7.44 | **1.28** | 13.71 | 22.43 |
| ViT-B/32 | `laion2b_s34b_b79k` | 67.72% | 87.56% | **3.23** | 1.67 | **13.11** | **18.02** |
| ViT-B/16 | `laion2b_s34b_b88k` | 69.20% | 87.71% | 3.11 | 1.72 | 16.91 | 21.74 |
| MobileCLIP2-S0 | `dfndr2b` | **73.91%** | **91.28%** | 4.75 | 1.68 | 15.54 | 21.97 |

Timing phases:
- **load** — model weights onto GPU
- **text** — zero-shot text embeddings for all 200 classes (13 prompts/class, mean-pooled)
- **eval** — forward pass over full 10,000-image validation set

**Key observations:**
- **TinyCLIP is competitive for its size.** With ~30% fewer vision params (61M vs ~87M) and 5× smaller pretraining data (LAION-400M vs LAION-2B), it trails ViT-B/32 by only 2.47 pp Top-1.
- **ViT-B/16 is best but slowest evaluator.** Smaller 16×16 patches produce ~4× more image tokens, making eval ~29% slower than ViT-B/32.
- **TinyCLIP text encoding is fastest.** Its smaller text encoder builds all class embeddings in 1.28s vs ~1.7s for full ViT-B variants.
- **TinyCLIP loads slower.** ~7.4s vs ~3.1–3.2s — a one-time cost per run attributed to HuggingFace Transformers loader vs. OpenCLIP native checkpoint loading.
- **Total wall-clock time is comparable.** TinyCLIP (22.4s) and ViT-B/16 (21.7s) are nearly identical end-to-end; ViT-B/32 is fastest overall (18.0s).

---

## Summary Comparison

| Model | Type | Vision Params | Train Data | Top-1 | Top-5 |
|---|---|---|---|---|---|
| TinyNet d4 | Supervised | ~small | Tiny ImageNet | 45.97% | 71.47% |
| ResNet-18 | Supervised | ~11M | Tiny ImageNet | 54.51% | 76.16% |
| ViT-B/32 (distilled student) | Distilled | half-width | Tiny ImageNet (teacher) | 34.70% | 58.09% |
| TinyCLIP ViT-61M-32 | Zero-shot | 61M | LAION-400M | 65.25% | 85.86% |
| ViT-B/32 | Zero-shot | ~87M | LAION-2B | 67.72% | 87.56% |
| ViT-B/16 | Zero-shot | ~87M | LAION-2B | 69.20% | 87.71% |
| MobileCLIP2-S0 | Zero-shot | 11.4M | DFN-2B | **73.91%** | **91.28%** |
| ViT-B/32 | Fine-tuned (k=3) | ~87M | Tiny ImageNet | 79.52% | 93.84% |

---

## Hardware & Software

- GPU: L40S or H200 (SLURM `--constraint="l40s|h200"`)
- Container: `cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif` + overlay `my_pytorch.ext3`
- Framework: OpenCLIP (`open_clip`), PyTorch with AMP enabled
- Batch size: 256, num_workers: 8

---

## Literature Review — Compact CLIP Models (2026-05-01)

Surveyed available compact CLIP-style models smaller than or comparable to TinyCLIP ViT-61M-32 (90M total params, 62.4% IN-1k zero-shot).
Motivation: identify a faster/cheaper stage-1 model for the 2-stage pipeline.

### TinyCLIP Family (Microsoft, ICCV 2023)

Distilled from larger CLIP teachers via affinity mimicking + weight inheritance. All variants on HuggingFace, loaded via `CLIPModel.from_pretrained()`.

| Model | Vision | Text | Total | IN-1k ZS | HuggingFace ID |
|---|---|---|---|---|---|
| ViT-61M/32 *(current)* | 61M | 29M | 90M | 62.4% | `wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M` |
| ViT-45M/32 | 45M | 18M | 63M | 61.4% | `wkcn/TinyCLIP-ViT-45M-32-Text-18M-LAION400M` |
| ViT-40M/32 | 40M | 19M | 59M | 59.8% | `wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M` |
| ViT-39M/16 | 39M | 19M | 58M | **63.5%** | `wkcn/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M` |
| ViT-22M/32 | 22M | 10M | 32M | 53.7% | HF available |
| ViT-8M/16 | 8M | 3M | 11M | 41.1% | `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` |

Notable: **ViT-39M/16** (YFCC-trained) is 36% smaller than the current model yet scores slightly *higher* on IN-1k (63.5% vs 62.4%).
The ViT-22M/32 drops 8.7 pp at 64% smaller — likely too steep for a viable stage-1 gate.

### MobileCLIP (Apple, CVPR 2024)

Multi-branch image encoder (MCi architecture, not ViT), distilled from ensemble teachers on DataCompDR-1B.
Available natively in `open_clip` — drop-in for existing code.

| Model | Vision | Text | Total | IN-1k ZS | open_clip name |
|---|---|---|---|---|---|
| **MobileCLIP-S0** | **11.4M** | **42.4M** | **53.8M** | **67.8%** | `MobileCLIP-S0`, `datacompdr` |
| MobileCLIP2-S0 | 11.4M | 63.4M | 74.8M | 71.5% | `MobileCLIP-S2`, `datacompdr` |
| MobileCLIP-S1 | 21.5M | 63.4M | 84.9M | 72.6% | `MobileCLIP-S1`, `datacompdr` |

MobileCLIP-S0 is the standout: **40% smaller than TinyCLIP ViT-61M-32 and 5.4 pp higher** on IN-1k zero-shot.
The image encoder (11.4M) is tiny — fast at inference. The text encoder (42.4M) is heavier than TinyCLIP's 29M but is computed once and cached, so it has no per-sample cost.

### SigLIP & EVA-CLIP

No small variants publicly available. Smallest released models are ViT-B scale (~149M total).
SigLIP ViT-B/16 achieves 76.2% IN-1k — strong accuracy but larger than TinyCLIP, so not useful as a smaller stage-1.
Available in both `open_clip` (`ViT-B-16-SigLIP`, `webli`) and HuggingFace (`google/siglip-base-patch16-224`).

### OpenCLIP ViT-S

No pretrained ViT-Small weights available in open_clip as of May 2026.

### Comparison vs Current Model

| Model | Total Params | IN-1k ZS | Δ Params | Δ Accuracy | Access |
|---|---|---|---|---|---|
| TinyCLIP ViT-61M/32 *(current)* | 90M | 62.4% | — | — | HF |
| TinyCLIP ViT-39M/16 (YFCC) | 58M | 63.5% | -36% | +1.1 pp | HF |
| TinyCLIP ViT-40M/32 | 59M | 59.8% | -34% | -2.6 pp | HF |
| TinyCLIP ViT-22M/32 | 32M | 53.7% | -64% | -8.7 pp | HF |
| MobileCLIP-S0 (V1) | 53.8M | 67.8% | -40% | +5.4 pp | open_clip — *not in installed version* |
| **MobileCLIP2-S0** | **74.8M** | **71.5% (IN-1k) / 73.91% (TinyIN)** | **-17%** | **+9.1 pp** | **open_clip (`dfndr2b`)** |
| SigLIP ViT-B/16 | ~149M | 76.2% | +66% | +13.8 pp | open_clip + HF |

### Recommendation

**Primary candidate for stage-1:** MobileCLIP2-S0 — available as `MobileCLIP2-S0 / dfndr2b` in open_clip. Note: `MobileCLIP-S0` (V1) is not in the installed open_clip version; `MobileCLIP2-S0` is the actual smallest available model.
**Secondary candidate:** TinyCLIP ViT-39M/16 (YFCC) — 36% smaller than current TinyCLIP, slightly better IN-1k, same HF loading path.

**Tested (job 7732506, 2026-05-01):** MobileCLIP2-S0 achieved **73.91% Top-1 / 91.28% Top-5** on Tiny ImageNet — beating ViT-B/16 (69.20%) by 4.71 pp at similar total wall-clock time.

---

## Next Steps — 2-Stage Pipeline

The immediate direction is building a 2-stage zero-shot classifier. Planned experiments:

### Step 1 — Per-class accuracy profiling
Run stage-1 model (TinyCLIP or ViT-B/32) and record per-class Top-1 accuracy across all 200 classes.
Identify which classes stage 1 consistently fails on — these are natural candidates for stage-2 routing.

### Step 2 — Define routing signal
Options under consideration (pick one to start):
- **Confidence threshold** — route samples where `max(softmax(logits)) < τ` to stage 2
- **Top-1/Top-5 disagreement** — if the top-1 and top-5 predictions are spread across many logits, route to stage 2
- **Per-class routing** — statically route all samples of known-hard classes to stage 2 (requires held-out calibration split)

### Step 3 — Measure the compute/accuracy trade-off
Track what fraction of samples get routed to stage 2 at each threshold, and the resulting combined Top-1.
Goal: find the operating point where stage-2 calls are minimal but accuracy is at or above full ViT-B/16 baseline (69.20%).
