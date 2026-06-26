# Decoupled Log-Polar Bias for Vision

> 🌐 日本語版: [`README_jp.md`](README_jp.md) ・ English version: this file

> A biologically-inspired positional encoding that extends KERPLE-log to 2D visual inputs.
> Positional-encoding comparison experiments on ViT-Tiny (multi-dataset).

---

## Results — CIFAR-100, ViT-Tiny, 300 epochs

| # | Method | Top-1 Acc | vs no_pe | PE params | Notes |
|---|--------|:---------:|:--------:|:---------:|-------|
| — | `resnet18` | — | — | — | ResNet baseline (from scratch) |
| — | `resnet50` | — | — | — | ResNet baseline (from scratch) |
| 1 | `no_pe` | 63.47% | — | 0 | No positional information |
| 2 | `ape` | 70.38% | +6.91pp | 12.3K | Learnable absolute PE |
| 3 | `alibi_2d` | 66.87% | +3.40pp | 0 | Linear distance bias, fixed slopes |
| 4 | `rpb` | 72.99% | +9.52pp | 8.1K | Swin V1 style, per-layer table |
| 5 | `cpb` | **76.61%** | +13.14pp | 36.9K | Swin V2 style, MLP |
| 6 | `rope_2d` | 74.67% | +11.20pp | 0 | RoPE2D baseline |
| 7 | `kerple_log_2d` | 69.89% | +6.42pp | 72 | Isotropic baseline (KERPLE-log extended to 2D) |
| 8 | **`dlpb`** | 73.65% | +10.18pp | 180 | **Proposed (anisotropic + per-head scale)** |
| 9 | **`dlpb_O2`** | 74.78% | +11.31pp | 252 | **Proposed (von Mises 2nd order + scale)** |
| 10 | **`dlpb_O3`** | 74.30% | +10.83pp | 324 | **Proposed (von Mises 3rd order + scale)** |
| 11 | **`dlpb_rope_2d`** | 74.26% | +10.79pp | 180 | **Proposed (anisotropic + RoPE2D hybrid)** |
| 12 | **`dlpb_O2_rope_2d`** | **75.32%** | +11.85pp | 252 | **Proposed (von Mises 2nd order + RoPE2D)** |
| 13 | **`dlpb_O3_rope_2d`** | 74.68% | +11.21pp | 324 | **Proposed (von Mises 3rd order + RoPE2D)** |

> PE params is the total of PE-module-specific parameters (depth=12 layers × params/layer). `python src/summarize_results.py cifar100` also prints the parameter-count table.
> The proposed methods are the 6 in rows 8–13 (`dlpb` / `dlpb_O2` / `dlpb_O3` and their `_rope_2d` hybrids).
> Other explored variants (non-scaled `dlpb_aniso` / `dlpb_vm`, `_sc_fix`, `dlpb_movm`) remain registered in `src/models.py` but are excluded from the default run/summary lists.

---

## Method overview

We inject the **foveated receptive fields** of the biological visual cortex (V1) into the ViT as an attention bias.

### KERPLE-log (`kerple_log_2d`) — isotropic

$$
\text{score}(i,j) = \frac{q_i^\top k_j}{\sqrt{d}} - r_1^{(h)} \log\!\left(1 + r_2^{(h)} \|p_j - p_i\|_2\right)
$$

### Anisotropic (`dlpb`) — adds orientation selectivity

$$
B_{ij}^{(h)} = -r_1^{(h)} \log\!\left(1 + r_2^{(h)} r_{ij}\right) + s_1^{(h)} \cos(\phi_{ij} - \phi_1^{*(h)})
$$

### Von Mises 2nd order (`dlpb_O2`) — edge selectivity (180° periodic)

$$
B_{ij}^{(h)} = -r_1^{(h)} \log\!\left(1 + r_2^{(h)} r_{ij}\right) + s_1^{(h)} \cos(\phi_{ij} - \phi_1^{*(h)}) + s_2^{(h)} \cos\!\left(2(\phi_{ij} - \phi_2^{*(h)})\right)
$$

The 2nd-order term is the 180°-periodic component corresponding to V1 orientation columns (edge/line selectivity).

### Von Mises 3rd order (`dlpb_O3`) — corner selectivity (120° periodic)

$$
B_{ij}^{(h)} = \underbrace{-r_1 \log(1+r_2 r)}_{\text{distance decay}} + \underbrace{s_1 \cos(\phi - \phi_1^*)}_{\text{360° direction}} + \underbrace{s_2 \cos 2(\phi - \phi_2^*)}_{\text{180° edge}} + \underbrace{s_3 \cos 3(\phi - \phi_3^*)}_{\text{120° corner}}
$$

The 3rd-order term captures local structures such as corners and three-way junctions.

---

## DLPB — Decoupled Log-Polar Bias

The proposed methods keep distance and orientation **decoupled** as separate additive terms: a logarithmic distance-decay term plus per-head von Mises angular terms, each scaled by a learnable per-head factor α (see "Method overview" above).

### Variant list

| PE name | Description | params/head |
|---------|-------------|:-----------:|
| `dlpb` | anisotropic (additive 1st) + per-head scale — **proposed** | 1.5 |
| `dlpb_O2` | von Mises 2nd order (additive 1st+2nd) + scale — **proposed** | 2.5 |
| `dlpb_O3` | von Mises 3rd order (additive 1st+2nd+3rd) + scale — **proposed** | 3.5 |
| `dlpb_{,_O2,_O3}_rope_2d` | above + RoPE2D hybrid — **proposed** | same |

**Core hypothesis**: the benefit of DLPB is inversely correlated with model scale.
On small models (ViT-Ti) it yields +3–10% improvement; on large models the effect vanishes.

- $r_1, r_2$: strength/scale of distance decay (per-head learnable, softplus-constrained)
- $\phi_k^{*(h)}$: preferred direction of head $h$ (learnable)
- $s_k^{(h)}$: strength of each angular term (learnable, softplus-constrained)

See [`log_polar_bias_for_vision.md`](log_polar_bias_for_vision.md) for details. For the DLPB design discussion, see [`DLPB_ellipse_anisotropy_summary.md`](DLPB_ellipse_anisotropy_summary.md).

---

## Method comparison

| Method | Injection site | Distance dependence | Orientation dependence | Receptive-field shape | Extrapolation | PE params |
|--------|----------------|---------------------|------------------------|-----------------------|---------------|:---------:|
| APE | token embedding | none | none | — | low | 12.3K |
| ALiBi-2D | attention logit | linear (fixed) | none | circle | medium | 0 |
| RPB | attention logit | learned (table) | learned | arbitrary | low | 8.1K |
| CPB | attention logit | learned (MLP) | learned | arbitrary | medium | 36.9K |
| RoPE2D (`rope_2d`) | Q/K rotation | relative rotation | relative rotation | circle | high | 0 |
| KERPLE-log 2D (`kerple_log_2d`) | attention logit | log | none | circle | high | 72 |
| **DLPB Aniso (`dlpb`)** | **attention logit** | **log** | **additive 1st** | **heart-shaped** | **high** | **180** |
| **DLPB VM (`dlpb_O2`)** | **attention logit** | **log** | **additive 1st+2nd** | **approx. ellipse** | **high** | **252** |
| **DLPB VM3 (`dlpb_O3`)** | **attention logit** | **log** | **additive 1st+2nd+3rd** | **approx. ellipse** | **high** | **324** |
| **DLPB *`_rope_2d`** | **both (logit + Q/K rotation)** | **log** | **additive + RoPE** | **approx. ellipse** | **high** | **same** |

> Bold rows are the proposed methods (`dlpb` / `dlpb_O2` / `dlpb_O3` and their RoPE2D hybrids `_rope_2d`).
> Other explored variants remain registered in `src/models.py` but are excluded from the default comparison list. See the "DLPB" section above.

---

## Supported datasets

| Dataset | Argument | Image size | Patch | Classes | Acquisition |
|---------|----------|:----------:|:-----:|:-------:|-------------|
| CIFAR-100 | `cifar100` | 32×32 | 4×4 | 100 | auto download |
| ImageNet-100 | `imagenet100` | 224×224 | 16×16 | 100 | manual placement required |
| Flowers-102 | `flowers102` | 224×224 | 16×16 | 102 | auto download |
| Stanford Cars | `cars` | 224×224 | 16×16 | 196 | manual placement required ※ |
| Oxford-IIIT Pets | `pets` | 224×224 | 16×16 | 37 | auto download |

> ※ The official Stanford Cars download link is down, so download it manually (e.g. from Kaggle) and place it under `data/stanford_cars/`.
> Place ImageNet-100 under `data/imagenet100/train/` and `data/imagenet100/val/` in ImageFolder layout.

---

## File layout

```
LPB-V/
├── README.md                     # English README (this file)
├── README_jp.md                  # Japanese README
├── log_polar_bias_for_vision.md  # research design / theory / experiment plan
├── DLPB_ellipse_anisotropy_summary.md  # DLPB design discussion
├── train.py                      # training script (all datasets; lives at repo root)
├── src/                          # Python sources other than train.py
│   ├── models.py                 # ViT-Tiny + all PE implementations + ResNet18/50
│   ├── eval_resolution.py        # resolution-extrapolation evaluation script
│   ├── summarize_results.py      # result aggregation / README update
│   ├── compare_cpb_dlpb.py       # bias comparison / visualization (CPB vs DLPB)
│   ├── visualize_pe.py           # PE bias visualization
│   ├── visualize_alpha.py        # per-head scale α visualization
│   └── plot_*.py                 # various figure generation
├── scripts/                      # shell scripts
│   ├── run_experiments.sh        # run all CIFAR-100 experiments
│   ├── run_imagenet100.sh / run_cars.sh / run_pets.sh / run_flowers102.sh
│   └── run_rope_2d.sh            # RoPE2D-family experiments
├── plot/                         # output directory for generated figures (.png)
├── data/
│   ├── cifar-100-python/         # CIFAR-100 (auto download)
│   ├── imagenet100/              # ImageNet-100 (manual)
│   │   ├── train/
│   │   └── val/
│   ├── stanford_cars/            # Stanford Cars (manual)
│   └── oxford-iiit-pet/          # Oxford Pets (auto download)
└── results/
    ├── resolution/
    │   └── {dataset}.json        # resolution-extrapolation results
    └── {dataset}/
        └── {pe_type}/
            ├── config.json       # experiment config
            ├── log.log           # per-epoch log
            ├── best.pth          # best model
            └── result.json       # final accuracy
```

> Run scripts from the repository root in principle (`train.py` is at the root, the rest live under `src/`). The relative paths inside `run_*.sh` (`./results`, `./data`, `train.py`) are also relative to the root.
> All generated figures (`.png`) are saved under `plot/`.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
(PyTorch 2.6.0+cu124).

```bash
# Create the virtual environment and install dependencies (.venv/)
uv sync

# Sanity check (forward pass of all PEs)
uv run python src/models.py
```

`uv run` automatically uses the project's `.venv`, so there is no environment to
activate manually. Prefix any command with `uv run` (e.g. `uv run python train.py ...`).

---

## Running experiments

### Run all CIFAR-100 experiments at once (including ResNet baselines)

```bash
bash scripts/run_experiments.sh
# ~6.5 hours on an RTX 4090 (cuda:1)
```

### Single experiment

```bash
# ViT + PE method
uv run python train.py --dataset cifar100   --pe_type dlpb_O2 --gpu 1 --epochs 300
uv run python train.py --dataset flowers102 --pe_type ape        --gpu 1 --epochs 200

# ResNet baseline
uv run python train.py --dataset cifar100   --pe_type resnet18 --gpu 1 --epochs 300
uv run python train.py --dataset imagenet100 --pe_type resnet50 --gpu 1 --epochs 300
```

### Main arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar100` | `cifar100 / imagenet100 / flowers102 / cars / pets` |
| `--pe_type` | `ape` | Proposed: `dlpb / dlpb_O2 / dlpb_O3` (+ `_rope_2d` hybrids)<br>Baselines: `no_pe / ape / alibi_2d / rpb / cpb / rope_2d / kerple_log_2d / resnet18 / resnet50`<br>Explored variants (registered only): `dlpb_aniso / dlpb_vm / dlpb_vm3 / dlpb_ape_vm / *_sc_fix / dlpb_movm*` (+ `_sc`, `_rope_2d`) |
| `--epochs` | `300` | number of training epochs |
| `--batch_size` | `256` | batch size |
| `--lr` | `5e-4` | peak learning rate (cosine decay) |
| `--gpu` | `0` | CUDA device index |
| `--data_dir` | `./data` | dataset root directory |
| `--results_dir` | `./results` | output directory for results |

### Checking results

```bash
uv run python src/summarize_results.py              # list available datasets
uv run python src/summarize_results.py cifar100     # show cifar100 results
```

### Resolution-extrapolation evaluation

Test at resolutions different from training and compare the resolution generalization of each PE type.

```bash
uv run python src/eval_resolution.py              # list available datasets
uv run python src/eval_resolution.py cifar100     # evaluate cifar100
uv run python src/eval_resolution.py flowers102   # evaluate flowers102
```

The evaluated PE types use the same list as `summarize_results.py` (8 baselines + 6 proposed methods). Individual selection is also possible via `--pe_types`.
Results are saved to `./results/resolution/{dataset}.json`.

Main arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset` | (auto-detected) | dataset name to evaluate |
| `--results_dir` | `./results` | where to search for checkpoints |
| `--data_dir` | `./data` | dataset root directory |
| `--gpu` | `0` | CUDA device index |
| `--batch_size` | `128` | batch size |

#### Test resolutions

| Dataset | Training resolution | Test resolutions |
|---------|:-------------------:|------------------|
| cifar100 | **32px** | 16, 24, **32**, 40, 48, 56, 64 |
| flowers102 / cars / pets | **224px** | 96, 128, 160, 192, **224**, 256, 320, 384 |

#### Resolution-adaptation scheme per PE type

| PE type | Adaptation scheme |
|---------|-------------------|
| `no_pe` | unchanged (no positional information) |
| `ape` | bilinearly interpolate pos_embed (2D reshape → interp → flatten) |
| `alibi_2d` | recompute the dist buffer on the new grid (slopes unchanged) |
| `rpb` | bilinearly interpolate the bias table, rebuild idx |
| `cpb` | recompute rel_coords on the new grid |
| `dlpb_*` | recompute dist / angle buffers (reuse $r_1, r_2, \phi^*, s$) |

> Because DLPB expresses distance decay with functional parameters ($r_1, r_2$, etc.), it generalizes to arbitrary resolutions by buffer recomputation alone.

#### CIFAR-100 example results (Top-1 Acc %)

| PE | 16px | 24px | **32px** | 40px | 48px | 56px | 64px |
|----|-----:|-----:|---------:|-----:|-----:|-----:|-----:|
| `no_pe` | 38.3 | 57.1 | 63.5 | 46.2 | 34.3 | 26.1 | 16.5 |
| `ape` | 46.6 | 64.8 | **70.4** | 56.4 | 44.5 | 37.2 | 27.4 |
| `alibi_2d` | 41.4 | 60.7 | 66.9 | 50.8 | 39.1 | 31.5 | 22.2 |
| `rpb` | 47.0 | 66.3 | 73.0 | 57.6 | 45.9 | 38.1 | 28.6 |
| `cpb` | 45.3 | 70.2 | **76.6** | 64.1 | 55.3 | 48.7 | 35.2 |
| `kerple_log_2d` | 39.1 | 63.1 | 69.9 | 53.8 | 41.9 | 34.6 | 24.7 |
| `dlpb_aniso` | 48.0 | 66.9 | 73.4 | 59.5 | 48.5 | 41.3 | 30.5 |
| `dlpb_vm` | 49.1 | 68.0 | 73.9 | 61.0 | **48.9** | 40.4 | 28.4 |
| `dlpb_vm3` | **49.7** | **68.1** | 74.1 | 60.6 | 48.5 | 40.2 | 29.3 |
| `dlpb_ape_vm` | **49.9** | 67.4 | 73.0 | **61.0** | 48.5 | 40.3 | 28.9 |

> The example above uses the previously-explored variants. Bold marks the best value at each resolution. Full results are saved to `results/resolution/{dataset}.json`.

---

## Model / training settings

### ViT-Tiny (CIFAR-100)

| Item | Value |
|------|-------|
| Input / patch | 32×32 / 4×4 (grid 8×8 = 64 tokens) |
| Embedding dim | 192 |
| Depth / heads | 12 / 3 |
| Pooling | Global Average Pooling |
| Parameters | ~5.4M |

### ViT-Tiny (224×224 datasets)

| Item | Value |
|------|-------|
| Input / patch | 224×224 / 16×16 (grid 14×14 = 196 tokens) |
| Embedding dim | 192 |
| Depth / heads | 12 / 3 |
| Pooling | Global Average Pooling |

### ResNet baselines

| Item | Value |
|------|-------|
| Training | from scratch (no pretrained weights) |
| CIFAR-100 adaptation | change the first conv from 7×7 stride-2 to 3×3 stride-1, replace maxpool with Identity |
| 224×224 datasets | standard ResNet configuration |

### Training settings (common to all models)

| Item | Value |
|------|-------|
| Optimizer | AdamW (β=0.9, 0.999) |
| LR schedule | Warmup 10 ep → Cosine decay |
| LR / min LR | 5e-4 / 1e-6 |
| Weight decay | 0.05 (0 for bias / norm / PE parameters) |
| Augmentation | CIFAR: RandomCrop + Flip / 224px: RandomResizedCrop + Flip |
| Mixup | α=0.8 |
| Label smoothing | 0.1 |
| Grad clip | max_norm=1.0 |

---

## References

- Chi et al. (2022). **KERPLE**: Kernelized Relative Positional Embedding for Length Extrapolation. *NeurIPS 2022*
- Press et al. (2022). **ALiBi**: Train Short, Test Long: Attention with Linear Biases. *ICLR 2022*
- Heo et al. (2024). Rotary Position Embedding for Vision Transformer. *ECCV 2024*
- Liu et al. (2022). **Swin Transformer V2**: Scaling Up Capacity and Resolution. *CVPR 2022*
- Touvron et al. (2021). **DeiT**: Training data-efficient image transformers. *ICML 2021*
- Fisher (1953). **von Mises distribution**: Dispersion on a Sphere. *Proc. Royal Society A*
  (theoretical background for the von Mises angular terms in `dlpb_O2` / `dlpb_O3`)
