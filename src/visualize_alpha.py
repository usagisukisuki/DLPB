"""
dlpb_O2 の alpha パラメータを可視化するスクリプト。

Usage:
  python visualize_alpha.py                        # best.pth を可視化
  python visualize_alpha.py --dataset flowers102   # 別データセット
  python visualize_alpha.py --epochs               # エポック推移も表示
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

RESULTS_ROOT = 'results'
PE_TYPE = 'dlpb_O2'

CHECKPOINT_EPOCHS = [50, 100, 150, 200, 250, 300]


def load_alpha(path):
    """チェックポイントから alpha テンソルを読み込む。形状: [n_blocks, n_heads]"""
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['model_state']
    keys = sorted([k for k in state if 'alpha' in k])
    return np.stack([state[k].numpy() for k in keys])  # [B, H]


def plot_heatmap(alpha, title, ax):
    n_blocks, n_heads = alpha.shape
    im = ax.imshow(alpha, cmap='RdBu_r', vmin=0, aspect='auto')
    ax.set_xlabel('Head')
    ax.set_ylabel('Block')
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels([f'B{i}' for i in range(n_blocks)])
    ax.set_title(title, fontsize=11)
    for b in range(n_blocks):
        for h in range(n_heads):
            ax.text(h, b, f'{alpha[b, h]:.2f}', ha='center', va='center',
                    fontsize=7, color='black')
    return im


def plot_bar(alpha, ax):
    n_blocks, n_heads = alpha.shape
    x = np.arange(n_blocks)
    width = 0.25
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for h in range(n_heads):
        ax.bar(x + h * width, alpha[:, h], width, label=f'Head {h}',
               color=colors[h], alpha=0.8)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='init (1.0)')
    ax.set_xlabel('Block')
    ax.set_ylabel('α')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'B{i}' for i in range(n_blocks)], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title('α per block and head (bar chart)', fontsize=11)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', alpha=0.3)


def plot_epoch_lines(dataset, ax):
    """各エポックのチェックポイントから alpha の平均を折れ線で示す。"""
    result_dir = os.path.join(RESULTS_ROOT, dataset, PE_TYPE)
    epochs_found, means, stds = [], [], []
    for ep in CHECKPOINT_EPOCHS:
        path = os.path.join(result_dir, f'ckpt_ep{ep}.pth')
        if os.path.exists(path):
            alpha = load_alpha(path)
            epochs_found.append(ep)
            means.append(alpha.mean())
            stds.append(alpha.std())

    best_path = os.path.join(result_dir, 'best.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location='cpu')
        best_ep = ckpt.get('epoch', None)
        best_alpha = load_alpha(best_path)
        if best_ep not in epochs_found:
            epochs_found.append(best_ep)
            means.append(best_alpha.mean())
            stds.append(best_alpha.std())

    if not epochs_found:
        ax.text(0.5, 0.5, 'No checkpoints found', ha='center', va='center',
                transform=ax.transAxes)
        return

    order = np.argsort(epochs_found)
    epochs_found = np.array(epochs_found)[order]
    means = np.array(means)[order]
    stds = np.array(stds)[order]

    ax.plot(epochs_found, means, marker='o', color='steelblue', linewidth=1.8)
    ax.fill_between(epochs_found, means - stds, means + stds, alpha=0.2, color='steelblue')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='init (1.0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('α  (mean ± std over all blocks & heads)')
    ax.set_title('α evolution over training', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100',
                        choices=['cifar100', 'flowers102', 'pets', 'cars', 'imagenet100'])
    parser.add_argument('--epochs', action='store_true',
                        help='エポック推移グラフを追加')
    args = parser.parse_args()

    result_dir = os.path.join(RESULTS_ROOT, args.dataset, PE_TYPE)
    best_path = os.path.join(result_dir, 'best.pth')
    if not os.path.exists(best_path):
        print(f'Checkpoint not found: {best_path}')
        return

    alpha = load_alpha(best_path)
    ckpt = torch.load(best_path, map_location='cpu')
    epoch = ckpt.get('epoch', '?')
    val_acc = ckpt.get('val_acc', None)
    title_suffix = f'  (epoch={epoch}, val_acc={val_acc:.2f}%)' if val_acc else f'  (epoch={epoch})'

    n_rows = 3 if args.epochs else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))
    fig.suptitle(f'{PE_TYPE} — α values  [{args.dataset}]{title_suffix}', fontsize=13)

    # ヒートマップ
    im = plot_heatmap(alpha, 'α heatmap  (block × head)', axes[0])
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)

    # バーチャート
    plot_bar(alpha, axes[1])

    if args.epochs:
        plot_epoch_lines(args.dataset, axes[2])

    plt.tight_layout()
    out_path = f'plot/alpha_{PE_TYPE}_{args.dataset}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
