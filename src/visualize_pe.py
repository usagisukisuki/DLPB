"""
Visualize positional encoding bias patterns for each PE type.

Usage:
  python visualize_pe.py                        # 全PE比較 (初期値)
  python visualize_pe.py --trained cifar100     # 学習済み重みで可視化
  python visualize_pe.py --pe dlpb_vm --trained cifar100 --detail  # 詳細表示
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from models import build_model, PE_TYPES


# ============================================================
# Load utilities
# ============================================================

def load_model(pe_type, dataset='cifar100', device='cpu'):
    """Load trained model checkpoint if available, else return fresh model."""
    configs = {
        'cifar100':   dict(image_size=32,  patch_size=4,  num_classes=100),
        'flowers102': dict(image_size=224, patch_size=16, num_classes=102),
        'cars':       dict(image_size=224, patch_size=16, num_classes=196),
        'pets':       dict(image_size=224, patch_size=16, num_classes=37),
    }
    cfg = configs[dataset]
    model = build_model(pe_type, **cfg)

    ckpt_path = f'results/{dataset}/{pe_type}/best.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        print(f'  Loaded: {ckpt_path}')
    else:
        print(f'  No checkpoint found for {pe_type}/{dataset}, using random init')

    model.eval()
    return model


# ============================================================
# Visualization helpers
# ============================================================

def get_all_layer_biases(model, N):
    """Return list of bias tensors [H, N, N] for each layer."""
    biases = []
    for block in model.blocks:
        b = block.attn.pe.get_attn_bias(N, 'cpu')
        if b is not None:
            biases.append(b.squeeze(0).detach())   # [H, N, N]
        else:
            biases.append(None)
    return biases


def bias_from_token(bias_HNN, token_idx, grid_size):
    """
    Extract the bias seen FROM a specific query token (row in N×N matrix),
    reshaped to [H, grid, grid].
    """
    H, N, _ = bias_HNN.shape
    row = bias_HNN[:, token_idx, :]          # [H, N]
    return row.reshape(H, grid_size, grid_size).numpy()


def center_token(grid_size):
    return (grid_size // 2) * grid_size + grid_size // 2


# ============================================================
# Plot: all PE types side-by-side (one row per PE type)
# ============================================================

def plot_all_pe(pe_types, dataset, trained, save_path='plot/pe_overview.png'):
    """
    Grid: rows = PE types, cols = attention heads (layer 0 bias from center token).
    """
    configs = {
        'cifar100':   dict(image_size=32,  patch_size=4),
        'flowers102': dict(image_size=224, patch_size=16),
    }
    cfg = configs.get(dataset, configs['cifar100'])
    grid_size = cfg['image_size'] // cfg['patch_size']
    N = grid_size ** 2
    num_heads = 3
    center = center_token(grid_size)

    # Filter to only PE types with attention bias
    bias_types = []
    for pe in pe_types:
        if pe in ('no_pe', 'ape', 'resnet18', 'resnet50'):
            continue
        bias_types.append(pe)

    n_rows = len(bias_types)
    n_cols = num_heads

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5),
                             squeeze=False)
    fig.suptitle(
        f'Attention Bias from Center Token  '
        f'({"trained: " + dataset if trained else "random init"}, layer 0)',
        fontsize=13, y=1.01
    )

    for row_i, pe_type in enumerate(bias_types):
        if trained:
            model = load_model(pe_type, dataset)
        else:
            model = build_model(pe_type,
                                image_size=cfg['image_size'],
                                patch_size=cfg['patch_size'],
                                num_classes=100)
            model.eval()

        biases = get_all_layer_biases(model, N)
        b0 = biases[0]   # [H, N, N]

        if b0 is None:
            for col in range(n_cols):
                axes[row_i][col].axis('off')
                axes[row_i][col].set_title('(no bias)')
            axes[row_i][0].set_ylabel(pe_type, rotation=0, labelpad=60, va='center')
            continue

        bias_from_center = bias_from_token(b0, center, grid_size)  # [H, g, g]
        vmax = np.abs(bias_from_center).max() or 1.0

        for h in range(n_cols):
            ax = axes[row_i][h]
            im = ax.imshow(bias_from_center[h],
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if row_i == 0:
                ax.set_title(f'Head {h}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[row_i][0].set_ylabel(pe_type, rotation=0, labelpad=80, va='center',
                                  fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.show()


# ============================================================
# Plot: all layers for a single PE type
# ============================================================

def plot_all_layers(pe_type, dataset, trained, save_path=None):
    """Rows = layers, cols = heads."""
    configs = {
        'cifar100':   dict(image_size=32,  patch_size=4),
        'flowers102': dict(image_size=224, patch_size=16),
    }
    cfg = configs.get(dataset, configs['cifar100'])
    grid_size = cfg['image_size'] // cfg['patch_size']
    N = grid_size ** 2
    num_heads = 3
    center = center_token(grid_size)

    if trained:
        model = load_model(pe_type, dataset)
    else:
        model = build_model(pe_type,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            num_classes=100)
        model.eval()

    biases = get_all_layer_biases(model, N)   # list of [H,N,N] or None
    bias_layers = [b for b in biases if b is not None]
    n_layers = len(bias_layers)
    if n_layers == 0:
        print(f'{pe_type} has no attention bias.')
        return

    fig, axes = plt.subplots(n_layers, num_heads,
                             figsize=(num_heads * 2.5, n_layers * 2.2),
                             squeeze=False)
    fig.suptitle(
        f'{pe_type}  —  bias from center token per layer  '
        f'({"trained: " + dataset if trained else "random init"})',
        fontsize=12, y=1.01
    )

    for li, bl in enumerate(bias_layers):
        bfc = bias_from_token(bl, center, grid_size)   # [H, g, g]
        vmax = np.abs(bfc).max() or 1.0
        for h in range(num_heads):
            ax = axes[li][h]
            im = ax.imshow(bfc[h], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(f'Head {h}', fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[li][0].set_ylabel(f'L{li}', rotation=0, labelpad=25, va='center')

    plt.tight_layout()
    sp = save_path or f'plot/{pe_type}_layers.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.show()


# ============================================================
# Plot: detail view for LPB-V (learned params + decay curve)
# ============================================================

def plot_dlpb_detail(pe_type, dataset, trained, save_path=None):
    """Show learned r1, r2, phi_star, and distance decay curve for LPB-V."""
    if 'dlpb' not in pe_type:
        print(f'{pe_type} is not an LPB-V variant.')
        return

    configs = {
        'cifar100':   dict(image_size=32,  patch_size=4),
        'flowers102': dict(image_size=224, patch_size=16),
    }
    cfg = configs.get(dataset, configs['cifar100'])
    grid_size = cfg['image_size'] // cfg['patch_size']
    num_heads = 3

    if trained:
        model = load_model(pe_type, dataset)
    else:
        model = build_model(pe_type,
                            image_size=cfg['image_size'],
                            patch_size=cfg['patch_size'],
                            num_classes=100)
        model.eval()

    import torch.nn.functional as F
    pe = model.blocks[0].attn.pe

    r1 = F.softplus(pe.r1_raw).detach().numpy()   # [H]
    r2 = F.softplus(pe.r2_raw).detach().numpy()   # [H]

    has_angle = hasattr(pe, 'phi_star')
    has_vm2   = hasattr(pe, 'phi_star2')
    has_vm3   = hasattr(pe, 'phi_star3')

    # Distance decay curves
    r_vals = np.linspace(0, grid_size * math.sqrt(2), 200)
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ---- Panel 1: decay curves per head ----
    ax1 = fig.add_subplot(gs[0])
    for h in range(num_heads):
        decay = -r1[h] * np.log1p(r2[h] * r_vals)
        ax1.plot(r_vals, decay, label=f'H{h}  r1={r1[h]:.3f} r2={r2[h]:.3f}')
    ax1.set_xlabel('Euclidean distance (patches)')
    ax1.set_ylabel('Bias value')
    ax1.set_title('Distance decay  -r1·log(1+r2·d)')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- Panel 2: learned r1, r2 per head across layers ----
    ax2 = fig.add_subplot(gs[1])
    r1s, r2s = [], []
    for block in model.blocks:
        p = block.attn.pe
        r1s.append(F.softplus(p.r1_raw).detach().numpy())
        r2s.append(F.softplus(p.r2_raw).detach().numpy())
    r1s = np.array(r1s)   # [L, H]
    r2s = np.array(r2s)

    for h in range(num_heads):
        ax2.plot(r1s[:, h], marker='o', ms=4, label=f'H{h} r1')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('r1 (distance weight)')
    ax2.set_title('r1 per layer × head')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ---- Panel 3: preferred directions phi* (polar plot) ----
    ax3 = fig.add_subplot(gs[2], projection='polar')
    if has_angle:
        phi_star = pe.phi_star.detach().numpy()   # [H]
        s = F.softplus(pe.s_raw).detach().numpy()
        ax3.scatter(phi_star, s, s=80, zorder=5)
        for h in range(num_heads):
            ax3.annotate(f'H{h}', (phi_star[h], s[h]), fontsize=8)
        ax3.set_title('Preferred direction φ* (r=strength)', pad=15)
    else:
        ax3.set_title('(isotropic — no angle params)')

    fig.suptitle(
        f'{pe_type} learned parameters  '
        f'({"trained: " + dataset if trained else "random init"})',
        fontsize=12
    )
    sp = save_path or f'plot/{pe_type}_detail.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.show()


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pe', default=None, help='Single PE type (default: all)')
    parser.add_argument('--dataset', default='cifar100',
                        choices=['cifar100', 'flowers102', 'cars', 'pets'])
    parser.add_argument('--trained', action='store_true',
                        help='Load trained checkpoint if available')
    parser.add_argument('--detail', action='store_true',
                        help='Detailed LPB-V parameter plots')
    parser.add_argument('--layers', action='store_true',
                        help='Show all layers for a single PE type')
    args = parser.parse_args()

    if args.pe:
        if args.detail:
            plot_dlpb_detail(args.pe, args.dataset, args.trained)
        elif args.layers:
            plot_all_layers(args.pe, args.dataset, args.trained)
        else:
            plot_all_pe([args.pe], args.dataset, args.trained,
                        save_path=f'plot/{args.pe}_overview.png')
    else:
        pe_types = [p for p in PE_TYPES if p not in ('no_pe', 'ape')]
        plot_all_pe(pe_types, args.dataset, args.trained)


if __name__ == '__main__':
    main()
