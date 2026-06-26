"""
Resolution OOD evaluation for trained LPB-V models.

Tests each trained model at multiple resolutions different from the training resolution,
showing how well each PE type generalises to unseen image sizes.

Adaptation strategy per PE type:
  NoPE        - works at any resolution unchanged
  APE         - bilinear interpolation of pos_embed (2-D reshape → interp → flatten)
  ALiBi2D     - recompute dist buffer for new grid (slopes are fixed/learned scalars)
  RPB         - bilinear interpolation of the (2g-1)² × H bias table; rebuild idx
  CPB         - recompute rel_coords for new grid
  LPBV_*      - recompute dist / angle buffers for new grid
                (r1,r2,phi_star,s parameters carry over directly — key LPBV advantage)
"""

import os
import sys
import argparse
import json
import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import (
    build_model, NoPE, APE, ALiBi2D, RPB, CPB,
    LPBVIsotropic, LPBVAnisotropic, LPBVVonMises, LPBVVonMisesV3,
    LPBVAnisotropicScaledFixed, LPBVVonMisesScaledFixed, LPBVVonMisesV3ScaledFixed,
    RoPE2D, RoPE2DHybrid,
    _make_dist, _make_angle, _make_rpb_idx, _make_cpb_coords,
)

# Baselines + the proposed methods. Other explored variants stay registered
# in models.py and can still be evaluated via --pe_types.
PE_TYPES = [
    'resnet18', 'no_pe', 'ape', 'alibi_2d', 'rpb', 'cpb', 'rope_2d', 'kerple_log_2d',
    'dlpb', 'dlpb_O2', 'dlpb_O3',
    'dlpb_rope_2d', 'dlpb_O2_rope_2d', 'dlpb_O3_rope_2d',
]

DATASET_CONFIGS = {
    'cifar100':    dict(num_classes=100, image_size=32,  patch_size=4),
    'flowers102':  dict(num_classes=102, image_size=224, patch_size=16),
    'cars':        dict(num_classes=196, image_size=224, patch_size=16),
    'pets':        dict(num_classes=37,  image_size=224, patch_size=16),
    'imagenet100': dict(num_classes=100, image_size=224, patch_size=16),
}


# ============================================================
# PE adaptation utilities
# ============================================================

def _adapt_ape(pe: APE, new_grid: int) -> None:
    """Bilinear-interpolate pos_embed from training grid to new_grid."""
    old_embed = pe.pos_embed.data           # [1, N_old, D]
    N_old = old_embed.shape[1]
    old_grid = int(round(math.sqrt(N_old)))
    if old_grid == new_grid:
        return
    D = old_embed.shape[2]
    # [1, D, g_old, g_old]
    embed_2d = old_embed.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    new_embed_2d = F.interpolate(embed_2d.float(), size=(new_grid, new_grid),
                                 mode='bilinear', align_corners=False)
    # [1, N_new, D]
    pe.pos_embed = nn.Parameter(
        new_embed_2d.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    )


def _adapt_alibi2d(pe: ALiBi2D, new_grid: int) -> None:
    new_dist = _make_dist(new_grid).to(pe.dist.device)
    pe.register_buffer('dist', new_dist)
    pe.grid_size = new_grid


def _adapt_rpb(pe: RPB, new_grid: int) -> None:
    """Bilinear-interpolate RPB table from (2*g_old-1)² to (2*g_new-1)², then rebuild idx."""
    old_grid = pe.idx.shape[0]
    if old_grid == int(round(math.sqrt(pe.idx.shape[0] * pe.idx.shape[1]))):
        # idx shape is [N_old, N_old]; determine old_grid from it
        old_grid = int(round(math.sqrt(pe.idx.shape[0])))
    else:
        old_grid = int(round(math.sqrt(pe.idx.shape[0])))

    if old_grid == new_grid:
        return

    num_heads = pe.table.shape[1]
    old_side = 2 * old_grid - 1
    new_side = 2 * new_grid - 1

    # table: [old_side², H]  →  2D: [H, old_side, old_side]
    tbl = pe.table.data.reshape(old_side, old_side, num_heads).permute(2, 0, 1).unsqueeze(0)  # [1,H,os,os]
    tbl_new = F.interpolate(tbl.float(), size=(new_side, new_side),
                            mode='bilinear', align_corners=False)   # [1,H,ns,ns]
    # back to [new_side², H]
    pe.table = nn.Parameter(tbl_new.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, num_heads))
    pe.register_buffer('idx', _make_rpb_idx(new_grid).to(pe.table.device))


def _adapt_cpb(pe: CPB, new_grid: int) -> None:
    new_coords = _make_cpb_coords(new_grid).to(pe.rel_coords.device)
    pe.register_buffer('rel_coords', new_coords)


def _adapt_rope_2d_2d(pe: RoPE2D, new_grid: int) -> None:
    """Recompute cos/sin buffers for a new grid size."""
    half = pe.cos.shape[-1] // 2
    d_pairs = half // 2
    base = 10000
    dev = pe.cos.device
    theta = 1.0 / (base ** (torch.arange(0, d_pairs).float() / d_pairs)).to(dev)
    pos = torch.arange(new_grid, device=dev).float()
    freqs = torch.outer(pos, theta)
    cos_f, sin_f = torch.cos(freqs), torch.sin(freqs)
    N = new_grid * new_grid
    rows = torch.arange(N, device=dev) // new_grid
    cols = torch.arange(N, device=dev) % new_grid
    cos_row = cos_f[rows].repeat_interleave(2, dim=-1)
    sin_row = sin_f[rows].repeat_interleave(2, dim=-1)
    cos_col = cos_f[cols].repeat_interleave(2, dim=-1)
    sin_col = sin_f[cols].repeat_interleave(2, dim=-1)
    pe.register_buffer('cos', torch.cat([cos_row, cos_col], dim=-1))
    pe.register_buffer('sin', torch.cat([sin_row, sin_col], dim=-1))


def _adapt_kerple_log_2d(pe: LPBVIsotropic, new_grid: int) -> None:
    pe.register_buffer('dist', _make_dist(new_grid).to(pe.dist.device))


def _adapt_dlpb_aniso(pe: LPBVAnisotropic, new_grid: int) -> None:
    dev = pe.dist.device
    pe.register_buffer('dist',  _make_dist(new_grid).to(dev))
    pe.register_buffer('angle', _make_angle(new_grid).to(dev))


def _adapt_dlpb_vm(pe: LPBVVonMises, new_grid: int) -> None:
    dev = pe.dist.device
    pe.register_buffer('dist',  _make_dist(new_grid).to(dev))
    pe.register_buffer('angle', _make_angle(new_grid).to(dev))


def _adapt_dlpb_vm3(pe: LPBVVonMisesV3, new_grid: int) -> None:
    dev = pe.dist.device
    pe.register_buffer('dist',  _make_dist(new_grid).to(dev))
    pe.register_buffer('angle', _make_angle(new_grid).to(dev))


def adapt_pe_module(pe, new_grid: int) -> None:
    if isinstance(pe, NoPE):
        pass
    elif isinstance(pe, APE):
        _adapt_ape(pe, new_grid)
    elif isinstance(pe, ALiBi2D):
        _adapt_alibi2d(pe, new_grid)
    elif isinstance(pe, RPB):
        _adapt_rpb(pe, new_grid)
    elif isinstance(pe, CPB):
        _adapt_cpb(pe, new_grid)
    elif isinstance(pe, LPBVVonMisesV3):
        _adapt_dlpb_vm3(pe, new_grid)
    elif isinstance(pe, LPBVVonMises):
        _adapt_dlpb_vm(pe, new_grid)
    elif isinstance(pe, LPBVAnisotropic):
        _adapt_dlpb_aniso(pe, new_grid)
    elif isinstance(pe, LPBVIsotropic):
        _adapt_kerple_log_2d(pe, new_grid)
    elif isinstance(pe, LPBVVonMisesV3ScaledFixed):
        _adapt_dlpb_vm3(pe, new_grid)
    elif isinstance(pe, LPBVVonMisesScaledFixed):
        _adapt_dlpb_vm(pe, new_grid)
    elif isinstance(pe, LPBVAnisotropicScaledFixed):
        _adapt_dlpb_aniso(pe, new_grid)
    elif isinstance(pe, RoPE2DHybrid):
        _adapt_rope_2d_2d(pe.rope, new_grid)
        adapt_pe_module(pe.bias_pe, new_grid)
    elif isinstance(pe, RoPE2D):
        _adapt_rope_2d_2d(pe, new_grid)


def adapt_model_to_grid(model: nn.Module, new_grid: int) -> nn.Module:
    """Return a copy of model with all PE buffers adapted to new_grid."""
    m = copy.deepcopy(model)
    m.grid_size = new_grid
    # Token-level APE
    if hasattr(m, 'ape') and m.ape is not None:
        _adapt_ape(m.ape, new_grid)
    # Per-layer attention-bias PE
    for block in m.blocks:
        adapt_pe_module(block.attn.pe, new_grid)
    return m


# ============================================================
# Evaluation helpers
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_val_loader(dataset: str, data_dir: str, image_size: int,
                   batch_size: int = 128, num_workers: int = 4) -> DataLoader:
    if dataset == 'cifar100':
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        ds = torchvision.datasets.CIFAR100(data_dir, train=False, transform=tf, download=True)

    elif dataset == 'flowers102':
        tf = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = torchvision.datasets.Flowers102(data_dir, split='val', transform=tf, download=True)

    elif dataset == 'cars':
        tf = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        cars_root = os.path.join(data_dir, 'stanford_cars')
        ds = torchvision.datasets.ImageFolder(os.path.join(cars_root, 'test'), transform=tf)

    elif dataset == 'pets':
        tf = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = torchvision.datasets.OxfordIIITPet(data_dir, split='test', transform=tf, download=True)

    elif dataset == 'imagenet100':
        tf = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'imagenet100', 'val'), transform=tf)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ============================================================
# Main
# ============================================================

# Test resolutions for each dataset (multiples of patch_size)
RESOLUTION_PLAN = {
    'cifar100': {
        'patch_size': 4,
        'train_size': 32,
        'test_sizes': [16, 32, 64, 128, 256],
    },
    'flowers102': {
        'patch_size': 16,
        'train_size': 224,
        'test_sizes': [56, 112, 224, 448, 768, 1024],
    },
    'cars': {
        'patch_size': 16,
        'train_size': 224,
        'test_sizes': [56, 112, 224, 448, 768, 1024],
    },
    'pets': {
        'patch_size': 16,
        'train_size': 224,
        'test_sizes': [56, 112, 224, 448, 768, 1024],
    },
    'imagenet100': {
        'patch_size': 16,
        'train_size': 224,
        'test_sizes': [96, 112, 224, 448, 768, 1024],
    },
}


def get_available_datasets(results_dir='./results'):
    if not os.path.isdir(results_dir):
        return []
    return sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d in RESOLUTION_PLAN
    )


def main():
    available = get_available_datasets()

    parser = argparse.ArgumentParser(description='Resolution OOD evaluation')
    parser.add_argument(
        'dataset',
        nargs='?',
        choices=available if available else None,
        default=available[0] if len(available) == 1 else None,
        help=f'Dataset to evaluate. Available: {available}',
    )
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pe_types',    nargs='+', default=None,
                        help='Evaluate only these PE types (default: all)')
    parser.add_argument('--only_sizes',  nargs='+', type=int, default=None,
                        help='Evaluate only these image sizes (default: all)')
    args = parser.parse_args()

    if args.dataset is None:
        if not available:
            print('No datasets found in ./results/')
            sys.exit(1)
        print(f'Available datasets: {available}')
        print(f'Usage: python eval_resolution.py <dataset>')
        sys.exit(0)

    dataset = args.dataset
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")

    plan = RESOLUTION_PLAN[dataset]
    ds_cfg = DATASET_CONFIGS[dataset]
    patch_size = plan['patch_size']
    train_size = plan['train_size']
    test_sizes = plan['test_sizes']
    if args.only_sizes:
        test_sizes = [s for s in test_sizes if s in args.only_sizes]

    pe_types_to_run = args.pe_types if args.pe_types else PE_TYPES

    ds_results_dir = os.path.join(args.results_dir, dataset)
    if not os.path.isdir(ds_results_dir):
        print(f"[ERROR] No results dir for {dataset}: {ds_results_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}  (train={train_size}px, patch={patch_size})")
    print(f"{'='*60}")

    # Load existing results to merge into (avoids re-running already-completed evals)
    out_dir = os.path.join('./results', 'resolution')
    out_path = os.path.join(out_dir, f'{dataset}.json')
    dataset_results = {}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            dataset_results = json.load(f)

    for pe_type in pe_types_to_run:
        ckpt_path = os.path.join(ds_results_dir, pe_type, 'best.pth')
        if not os.path.isfile(ckpt_path):
            print(f"  [{pe_type}] No checkpoint found, skipping.")
            continue

        # Skip ResNet baselines (not ViT — PE concept doesn't apply)
        if pe_type in ('resnet18', 'resnet50'):
            print(f"  [{pe_type}] ResNet baseline — skipping resolution OOD.")
            continue

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        train_args = ckpt.get('args', {})

        # Build model and load weights
        model = build_model(
            pe_type=pe_type,
            image_size=train_size,
            patch_size=patch_size,
            num_classes=ds_cfg['num_classes'],
            drop_path_rate=train_args.get('drop_path', 0.1),
        )
        model.load_state_dict(ckpt['model_state'])
        model.to(device)

        pe_results = dict(dataset_results.get(pe_type, {}))
        print(f"\n  [{pe_type}]  (train acc from ckpt: {ckpt.get('val_acc', '?'):.2f}%)")

        for img_size in test_sizes:
            if img_size % patch_size != 0:
                continue
            key = str(img_size)
            if pe_results.get(key) is not None:
                print(f"    {img_size:4d}px  already evaluated ({pe_results[key]:.2f}%), skipping.")
                continue
            new_grid = img_size // patch_size

            adapted = adapt_model_to_grid(model, new_grid)
            adapted.to(device)

            try:
                loader = get_val_loader(dataset, args.data_dir, img_size,
                                        args.batch_size, args.num_workers)
                acc = evaluate(adapted, loader, device)
                marker = " <-- train" if img_size == train_size else ""
                print(f"    {img_size:4d}px  grid {new_grid:2d}×{new_grid:<2d}  acc={acc:.2f}%{marker}")
                pe_results[key] = acc
            except Exception as e:
                print(f"    {img_size:4d}px  ERROR: {e}")
                pe_results[key] = None

            del adapted
            torch.cuda.empty_cache()

        dataset_results[pe_type] = pe_results

    # Print summary table
    print(f"\n  Summary table ({dataset})")
    header_sizes = [s for s in test_sizes if s % patch_size == 0]
    header = f"  {'PE':15s}" + "".join(f"  {s:>5d}" for s in header_sizes)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pe in dataset_results:
        row = f"  {pe:15s}"
        for s in header_sizes:
            v = dataset_results[pe].get(str(s))
            row += f"  {'N/A':>5s}" if v is None else f"  {v:5.1f}"
        print(row)

    # Save results (merge with existing)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(dataset_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
