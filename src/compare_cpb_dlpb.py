"""
CPB vs LPB-V learned bias comparison visualization.

Usage:
  python compare_cpb_dlpb.py                          # cifar100 デフォルト
  python compare_cpb_dlpb.py --dataset flowers102
  python compare_cpb_dlpb.py --dataset all            # 全データセット比較
  python compare_cpb_dlpb.py --dataset cifar100 --layers  # 全レイヤー表示
"""

import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model

DATASET_CONFIGS = {
    'cifar100':   dict(image_size=32,  patch_size=4,  num_classes=100),
    'flowers102': dict(image_size=224, patch_size=16, num_classes=102),
    'cars':       dict(image_size=224, patch_size=16, num_classes=196),
    'pets':       dict(image_size=224, patch_size=16, num_classes=37),
    'imagenet100': dict(image_size=224, patch_size=16, num_classes=100),
}

LPBV_VARIANTS = ['kerple_log_2d', 'dlpb_aniso', 'dlpb_vm', 'dlpb_vm3',
                 'dlpb', 'dlpb_O3']


# ============================================================
# Load
# ============================================================

def load_model(pe_type, dataset, device='cpu'):
    cfg = DATASET_CONFIGS[dataset]
    model = build_model(pe_type, **cfg)
    ckpt_path = f'results/{dataset}/{pe_type}/best.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        print(f'  Loaded {ckpt_path}')
    else:
        print(f'  [WARN] No checkpoint: {ckpt_path}')
    model.eval()
    return model


# ============================================================
# Bias extraction helpers
# ============================================================

def get_bias_layer(model, layer_idx, N):
    """[H, N, N] bias for a given layer (or None)."""
    b = model.blocks[layer_idx].attn.pe.get_attn_bias(N, 'cpu')
    return b.squeeze(0).detach() if b is not None else None


def center_token(grid_size):
    return (grid_size // 2) * grid_size + grid_size // 2


def bias_map(bias_HNN, token_idx, grid_size):
    """Return [H, g, g] bias map from a query token."""
    row = bias_HNN[:, token_idx, :]
    return row.reshape(-1, grid_size, grid_size).numpy()


def radial_profile(bias_2d, grid_size):
    """Rotationally averaged radial profile of a 2D bias map."""
    cy, cx = grid_size // 2, grid_size // 2
    ys, xs = np.mgrid[0:grid_size, 0:grid_size]
    r = np.sqrt((ys - cy)**2 + (xs - cx)**2).flatten()
    v = bias_2d.flatten()
    r_bins = np.arange(0, grid_size * math.sqrt(2), 0.5)
    means = []
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        means.append(v[mask].mean() if mask.sum() > 0 else np.nan)
    return r_bins[:-1], np.array(means)


# ============================================================
# LPB-V parameter extraction
# ============================================================

def extract_dlpb_params(model):
    """Return dict of per-layer LPB-V parameter arrays."""
    results = {'r1': [], 'r2': [], 's1': [], 'phi_star': [],
               's2': [], 'phi_star2': [], 's3': [], 'phi_star3': [],
               'alpha': []}
    for block in model.blocks:
        pe = block.attn.pe
        results['r1'].append(F.softplus(pe.r1_raw).detach().numpy())
        results['r2'].append(F.softplus(pe.r2_raw).detach().numpy())
        if hasattr(pe, 's_raw'):
            results['s1'].append(F.softplus(pe.s_raw).detach().numpy())
            results['phi_star'].append(pe.phi_star.detach().numpy())
        if hasattr(pe, 's_raw2'):
            results['s2'].append(F.softplus(pe.s_raw2).detach().numpy())
            results['phi_star2'].append(pe.phi_star2.detach().numpy())
        if hasattr(pe, 's_raw3'):
            results['s3'].append(F.softplus(pe.s_raw3).detach().numpy())
            results['phi_star3'].append(pe.phi_star3.detach().numpy())
        if hasattr(pe, 'alpha'):
            results['alpha'].append(pe.alpha.detach().numpy())
    for k in results:
        if results[k]:
            results[k] = np.array(results[k])  # [L, H]
    return results


# ============================================================
# Figure 1: CPB vs dlpb_vm — bias heatmaps (layer 0 & 6, all heads)
# ============================================================

def plot_bias_comparison(dataset, save_path=None,
                         pe_filter=None, check_layers=None, display_names=None):
    """
    pe_filter:     list of pe_type strings to include (None = all)
    check_layers:  list of layer indices to show (None = [0,3,6,9,11])
    display_names: dict mapping pe_type -> label string (None = use pe_type)
    """
    cfg = DATASET_CONFIGS[dataset]
    grid_size = cfg['image_size'] // cfg['patch_size']
    N = grid_size ** 2
    num_heads = 3
    ct = center_token(grid_size)
    if check_layers is None:
        check_layers = [0, 3, 6, 9, 11]

    pe_types = pe_filter if pe_filter is not None else [
        'cpb', 'kerple_log_2d', 'dlpb_aniso', 'dlpb_vm', 'dlpb_vm3',
        'dlpb', 'dlpb_O3']
    available = [p for p in pe_types
                 if os.path.exists(f'results/{dataset}/{p}/best.pth')]
    if not available:
        print(f'No checkpoints for {dataset}')
        return

    n_rows = len(available)
    n_cols = num_heads * len(check_layers)

    labels = [(display_names or {}).get(p, p) for p in available]
    # labelpad: ~6pt per char at fontsize 9, plus a small margin
    labelpad = max(len(l) for l in labels) * 6 + 8

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 2.0),
        squeeze=False, layout='constrained'
    )

    for ri, pe_type in enumerate(available):
        model = load_model(pe_type, dataset)
        for li, layer in enumerate(check_layers):
            b = get_bias_layer(model, layer, N)
            for h in range(num_heads):
                ci = li * num_heads + h
                ax = axes[ri][ci]
                if b is not None:
                    bmap = bias_map(b, ct, grid_size)[h]
                    vmin, vmax = float(bmap.min()), float(bmap.max())
                    im = ax.imshow(bmap, cmap='viridis',
                                   vmin=vmin, vmax=vmax,
                                   interpolation='nearest')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                 format='%.1f')
                    ax.set_xlabel(f'[{vmin:.1f}, {vmax:.1f}]', fontsize=5,
                                  labelpad=1)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f'L{layer}\nH{h}', fontsize=7)
        axes[ri][0].set_ylabel(labels[ri], rotation=0, labelpad=labelpad,
                               va='center', fontsize=9)

    sp = save_path or f'plot/comparison_heatmap_{dataset}.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.close()


# ============================================================
# Figure 2: Radial decay profiles CPB vs LPB-V
# ============================================================

def plot_radial_profiles(dataset, save_path=None):
    cfg = DATASET_CONFIGS[dataset]
    grid_size = cfg['image_size'] // cfg['patch_size']
    N = grid_size ** 2
    num_heads = 3
    ct = center_token(grid_size)

    pe_types = ['cpb', 'kerple_log_2d', 'dlpb_aniso', 'dlpb_vm', 'dlpb_vm3',
                'dlpb', 'dlpb_O3']
    available = [p for p in pe_types
                 if os.path.exists(f'results/{dataset}/{p}/best.pth')]

    # Layers to compare: early / mid / late
    check_layers = [0, 5, 11]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(
        num_heads, len(check_layers),
        figsize=(len(check_layers) * 4.5, num_heads * 3.0),
        squeeze=False
    )
    fig.suptitle(
        f'Radially-averaged bias profile  [{dataset}]\n'
        f'(rotational average from center token)',
        fontsize=12
    )

    for pi, pe_type in enumerate(available):
        model = load_model(pe_type, dataset)
        for li, layer in enumerate(check_layers):
            b = get_bias_layer(model, layer, N)
            for h in range(num_heads):
                ax = axes[h][li]
                if li == 0:
                    ax.set_ylabel(f'Head {h}  — bias', fontsize=9)
                if h == 0:
                    ax.set_title(f'Layer {layer}', fontsize=10)
                if b is not None:
                    bmap = bias_map(b, ct, grid_size)[h]
                    r, v = radial_profile(bmap, grid_size)
                    ax.plot(r, v, color=colors[pi % len(colors)],
                            label=pe_type, linewidth=1.5)
                ax.set_xlabel('Distance (patches)')
                ax.grid(alpha=0.3)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', fontsize=9,
                   bbox_to_anchor=(1.12, 0.95))

    plt.tight_layout()
    sp = save_path or f'plot/comparison_radial_{dataset}.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.close()


# ============================================================
# Figure 3: LPB-V learned parameters (r1, r2, s, phi*)
# ============================================================

def plot_dlpb_params(dataset, save_path=None):
    cfg = DATASET_CONFIGS[dataset]
    num_heads = 3
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']

    available = [p for p in LPBV_VARIANTS
                 if os.path.exists(f'results/{dataset}/{p}/best.pth')]
    if not available:
        print(f'No LPB-V checkpoints for {dataset}')
        return

    n_variants = len(available)
    fig = plt.figure(figsize=(16, 4 * n_variants))
    outer_gs = gridspec.GridSpec(n_variants, 1, figure=fig, hspace=0.5)

    for vi, pe_type in enumerate(available):
        model = load_model(pe_type, dataset)
        params = extract_dlpb_params(model)
        n_layers = params['r1'].shape[0]
        layers = np.arange(n_layers)

        has_alpha = len(params['alpha']) > 0
        n_panels = 5 if has_alpha else 4
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_panels, subplot_spec=outer_gs[vi], wspace=0.4
        )

        # -- r1 across layers --
        ax1 = fig.add_subplot(inner_gs[0])
        for h in range(num_heads):
            ax1.plot(layers, params['r1'][:, h], color=colors[h],
                     marker=markers[h], ms=4, label=f'H{h}')
        ax1.set_title('r1 (distance weight)', fontsize=9)
        ax1.set_xlabel('Layer'); ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)

        # -- r2 across layers --
        ax2 = fig.add_subplot(inner_gs[1])
        for h in range(num_heads):
            ax2.plot(layers, params['r2'][:, h], color=colors[h],
                     marker=markers[h], ms=4, label=f'H{h}')
        ax2.set_title('r2 (scale param)', fontsize=9)
        ax2.set_xlabel('Layer'); ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)

        # -- angular strengths s1, s2, s3 --
        ax3 = fig.add_subplot(inner_gs[2])
        has_s = [('s1', params['s1'], 'solid'),
                 ('s2', params['s2'], 'dashed'),
                 ('s3', params['s3'], 'dotted')]
        for sname, sarr, ls in has_s:
            if len(sarr) == 0:
                continue
            for h in range(num_heads):
                ax3.plot(layers, sarr[:, h], color=colors[h],
                         linestyle=ls, ms=3, marker=markers[h],
                         label=f'H{h} {sname}')
        ax3.set_title('Angular strengths s1/s2/s3', fontsize=9)
        ax3.set_xlabel('Layer'); ax3.grid(alpha=0.3)
        if ax3.get_lines():
            ax3.legend(fontsize=7, ncol=2)
        else:
            ax3.text(0.5, 0.5, '(isotropic)', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=9, color='gray')

        # -- polar: preferred directions at layer 0 --
        ax4 = fig.add_subplot(inner_gs[3], projection='polar')
        pe0 = model.blocks[0].attn.pe
        legend_items = []
        if len(params['phi_star']) > 0:
            phi1 = params['phi_star'][0]   # [H]
            s1v  = params['s1'][0]
            for h in range(num_heads):
                sc = ax4.scatter(phi1[h], s1v[h], color=colors[h],
                                 s=80, zorder=5, marker=markers[h])
                ax4.plot([0, phi1[h]], [0, s1v[h]], color=colors[h],
                         alpha=0.4, linewidth=1)
            legend_items.append(('1st-order φ*', 'solid'))
        if len(params['phi_star2']) > 0:
            phi2 = params['phi_star2'][0]
            s2v  = params['s2'][0]
            for h in range(num_heads):
                ax4.scatter(phi2[h], s2v[h], color=colors[h],
                            s=80, zorder=5, marker='x', linewidths=2)
                # 180°-periodic: plot both directions
                ax4.scatter(phi2[h] + math.pi, s2v[h], color=colors[h],
                            s=40, zorder=4, marker='x', linewidths=1, alpha=0.4)
        ax4.set_title('φ* at layer 0\n(r = strength)', fontsize=9, pad=15)

        # -- alpha scale parameter (sc variants only) --
        if has_alpha:
            ax5 = fig.add_subplot(inner_gs[4])
            for h in range(num_heads):
                ax5.plot(layers, params['alpha'][:, h], color=colors[h],
                         marker=markers[h], ms=4, label=f'H{h}')
            ax5.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
            ax5.set_title('α (scale)', fontsize=9)
            ax5.set_xlabel('Layer'); ax5.grid(alpha=0.3)
            ax5.legend(fontsize=8)

        fig.text(
            0.01, 1 - (vi + 0.5) / n_variants,
            pe_type, ha='left', va='center',
            fontsize=11, fontweight='bold',
            transform=fig.transFigure,
            rotation=90
        )

    fig.suptitle(f'LPB-V learned parameters  [{dataset}]', fontsize=13)
    sp = save_path or f'plot/comparison_dlpb_params_{dataset}.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.close()


# ============================================================
# Figure 4: Cross-dataset comparison of learned r1/r2/s for dlpb_vm
# ============================================================

def plot_cross_dataset(pe_type='dlpb_vm', save_path=None):
    datasets = [d for d in DATASET_CONFIGS
                if os.path.exists(f'results/{d}/{pe_type}/best.pth')]
    if not datasets:
        print(f'No checkpoints for {pe_type}')
        return

    num_heads = 3
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    markers = ['o', 's', '^']
    ds_colors = plt.cm.Set2.colors

    fig, axes = plt.subplots(3, len(datasets),
                             figsize=(len(datasets) * 4, 9),
                             squeeze=False)
    fig.suptitle(f'{pe_type} — cross-dataset parameter comparison', fontsize=12)

    for di, dataset in enumerate(datasets):
        cfg = DATASET_CONFIGS[dataset]
        model = load_model(pe_type, dataset)
        params = extract_dlpb_params(model)
        n_layers = params['r1'].shape[0]
        layers = np.arange(n_layers)

        ax0 = axes[0][di]
        ax1 = axes[1][di]
        ax2 = axes[2][di]

        ax0.set_title(dataset, fontsize=10)

        for h in range(num_heads):
            ax0.plot(layers, params['r1'][:, h], color=colors[h],
                     marker=markers[h], ms=4, label=f'H{h}')
            ax1.plot(layers, params['r2'][:, h], color=colors[h],
                     marker=markers[h], ms=4, label=f'H{h}')
            if len(params['s1']) > 0:
                ax2.plot(layers, params['s1'][:, h], color=colors[h],
                         linestyle='solid', marker=markers[h], ms=4, label=f'H{h} s1')
            if len(params['s2']) > 0:
                ax2.plot(layers, params['s2'][:, h], color=colors[h],
                         linestyle='dashed', marker=markers[h], ms=3,
                         alpha=0.7, label=f'H{h} s2')

        if di == 0:
            ax0.set_ylabel('r1'); ax1.set_ylabel('r2'); ax2.set_ylabel('s (angular)')
        ax0.grid(alpha=0.3); ax0.legend(fontsize=7)
        ax1.grid(alpha=0.3)
        ax2.grid(alpha=0.3)
        ax2.set_xlabel('Layer')

    plt.tight_layout()
    sp = save_path or f'plot/comparison_cross_dataset_{pe_type}.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.close()


# ============================================================
# Figure 5: CPB vs dlpb_vm — per-head bias difference
# ============================================================

def plot_bias_diff(dataset, dlpb_pe='dlpb_vm', save_path=None):
    """Show CPB bias, dlpb bias, and their difference side-by-side."""
    cfg = DATASET_CONFIGS[dataset]
    grid_size = cfg['image_size'] // cfg['patch_size']
    N = grid_size ** 2
    num_heads = 3
    ct = center_token(grid_size)
    check_layers = [0, 5, 11]

    if not (os.path.exists(f'results/{dataset}/cpb/best.pth') and
            os.path.exists(f'results/{dataset}/{dlpb_pe}/best.pth')):
        print(f'Missing checkpoints for diff plot [{dataset}] (cpb vs {dlpb_pe})')
        return

    cpb_model  = load_model('cpb',    dataset)
    dlpb_model = load_model(dlpb_pe,  dataset)

    # Layout per layer row:
    #   [CPB raw H0..H2] [LPB-V raw H0..H2] [CPB centered H0..H2] [LPB-V centered H0..H2] [diff H0..H2]
    n_rows = len(check_layers)
    n_cols = num_heads * 5

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 2.4),
                             squeeze=False)
    fig.suptitle(
        f'CPB vs {dlpb_pe}  [{dataset}]\n'
        f'Raw (viridis, per-map scale) | Zero-mean (RdBu_r) | Diff of zero-mean',
        fontsize=11, y=1.01
    )

    # Column group headers
    def _set_col_title(col, text):
        axes[0][col].set_title(text, fontsize=7)

    for h in range(num_heads):
        _set_col_title(h,                  f'CPB raw\nH{h}')
        _set_col_title(num_heads + h,      f'LPB-V raw\nH{h}')
        _set_col_title(2*num_heads + h,    f'CPB centered\nH{h}')
        _set_col_title(3*num_heads + h,    f'LPB-V centered\nH{h}')
        _set_col_title(4*num_heads + h,    f'Diff\nH{h}')

    for li, layer in enumerate(check_layers):
        bc = get_bias_layer(cpb_model,  layer, N)
        bl = get_bias_layer(dlpb_model, layer, N)

        cpb_maps  = bias_map(bc, ct, grid_size) if bc is not None else None
        dlpb_maps = bias_map(bl, ct, grid_size) if bl is not None else None

        axes[li][0].set_ylabel(f'Layer {layer}', rotation=0, labelpad=55,
                               va='center', fontsize=9)

        for h in range(num_heads):
            def _raw(ax, data):
                """Natural scale with viridis — works for all-positive or all-negative."""
                vmin, vmax = float(data.min()), float(data.max())
                im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax,
                               interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1f')
                ax.set_xticks([]); ax.set_yticks([])

            def _centered(ax, data):
                """Zero-mean so spatial structure is visible regardless of sign."""
                d = data - data.mean()
                vmax = max(abs(d.max()), abs(d.min()), 1e-6)
                im = ax.imshow(d, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                               interpolation='nearest')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.2f')
                ax.set_xticks([]); ax.set_yticks([])

            if cpb_maps is not None:
                _raw(axes[li][h], cpb_maps[h])
                _centered(axes[li][2*num_heads + h], cpb_maps[h])
            if dlpb_maps is not None:
                _raw(axes[li][num_heads + h], dlpb_maps[h])
                _centered(axes[li][3*num_heads + h], dlpb_maps[h])
            if cpb_maps is not None and dlpb_maps is not None:
                c = cpb_maps[h]
                l = dlpb_maps[h]
                c_n = (c - c.mean()) / (c.std() + 1e-8)
                l_n = (l - l.mean()) / (l.std() + 1e-8)
                _centered(axes[li][4*num_heads + h], c_n - l_n)

    plt.tight_layout()
    sp = save_path or f'plot/comparison_diff_{dataset}_{dlpb_pe}.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    print(f'Saved: {sp}')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100',
                        choices=list(DATASET_CONFIGS.keys()) + ['all'])
    parser.add_argument('--layers', action='store_true',
                        help='Also generate all-layers view per PE')
    args = parser.parse_args()

    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]

    # LPB-V variants to use for diff plots (cpb vs each)
    diff_variants = ['dlpb_vm', 'dlpb_vm3', 'dlpb', 'dlpb_O3']

    for ds in datasets:
        print(f'\n=== {ds} ===')
        print('-- Bias heatmaps --')
        plot_bias_comparison(ds)

        print('-- Radial profiles --')
        plot_radial_profiles(ds)

        print('-- LPB-V parameter plots --')
        plot_dlpb_params(ds)

        print('-- CPB vs LPB-V diff --')
        for variant in diff_variants:
            plot_bias_diff(ds, dlpb_pe=variant)

    print('\n-- Cross-dataset dlpb_vm --')
    plot_cross_dataset('dlpb_vm')

    print('\n-- Cross-dataset dlpb_vm3 --')
    plot_cross_dataset('dlpb_vm3')

    print('\n-- Cross-dataset dlpb --')
    plot_cross_dataset('dlpb')

    print('\n-- Cross-dataset dlpb_O3 --')
    plot_cross_dataset('dlpb_O3')

    print('\nDone.')


if __name__ == '__main__':
    main()
