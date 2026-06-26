"""
Training script for LPB-V position encoding comparison.

Supported datasets:
  cifar100    - CIFAR-100 (32x32, 100 classes) -- auto-downloaded
  imagenet100 - ImageNet-100 (224x224, 100 classes) -- provide data at data_dir/imagenet100/{train,val}
  flowers102  - Flowers-102 (224x224, 102 classes) -- auto-downloaded
  cars        - Stanford Cars (224x224, 196 classes) -- manual download required (broken upstream)
  pets        - Oxford-IIIT Pets (224x224, 37 classes) -- auto-downloaded

Usage:
  python train.py --dataset cifar100    --pe_type ape --epochs 300 --gpu 0
  python train.py --dataset imagenet100 --pe_type kerple_log_2d --epochs 300 --gpu 0
  python train.py --dataset flowers102  --pe_type dlpb_vm --epochs 200 --gpu 0
"""

import os
import sys
import time
import math
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from models import build_model, count_params, MODEL_TYPES, BASELINE_MODELS


# ============================================================
# Dataset registry
# ============================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DATASET_CONFIGS = {
    'cifar100': dict(
        num_classes=100, image_size=32, patch_size=4,
        mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761),
    ),
    'imagenet100': dict(
        num_classes=100, image_size=224, patch_size=16,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
    'flowers102': dict(
        num_classes=102, image_size=224, patch_size=16,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
    'cars': dict(
        num_classes=196, image_size=224, patch_size=16,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
    'pets': dict(
        num_classes=37, image_size=224, patch_size=16,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    ),
}


def _train_tf_large(image_size, mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def _val_tf_large(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_loaders(dataset, data_dir, batch_size, num_workers=8):
    cfg = DATASET_CONFIGS[dataset]
    img_size = cfg['image_size']
    mean, std = cfg['mean'], cfg['std']

    if dataset == 'cifar100':
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_ds = torchvision.datasets.CIFAR100(data_dir, train=True,  transform=train_tf, download=True)
        val_ds   = torchvision.datasets.CIFAR100(data_dir, train=False, transform=val_tf,   download=True)

    elif dataset == 'imagenet100':
        # Expects data at data_dir/imagenet100/train and data_dir/imagenet100/val (ImageFolder layout)
        train_tf = _train_tf_large(img_size, mean, std)
        val_tf   = _val_tf_large(img_size, mean, std)
        train_ds = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'imagenet100', 'train'), transform=train_tf)
        val_ds   = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'imagenet100', 'val'),   transform=val_tf)

    elif dataset == 'flowers102':
        # Training on 'train' split, evaluation on 'val' split.
        # For the full benchmark (train+val → test), adjust splits here.
        train_tf = _train_tf_large(img_size, mean, std)
        val_tf   = _val_tf_large(img_size, mean, std)
        train_ds = torchvision.datasets.Flowers102(data_dir, split='train', transform=train_tf, download=True)
        val_ds   = torchvision.datasets.Flowers102(data_dir, split='val',   transform=val_tf,   download=True)

    elif dataset == 'cars':
        # Stanford Cars (class-folder layout from Kaggle jutrera/stanford-car-dataset-by-classes-folder).
        # Place data so that data_dir/stanford_cars/{train,test}/<class_name>/*.jpg exist.
        train_tf = _train_tf_large(img_size, mean, std)
        val_tf   = _val_tf_large(img_size, mean, std)
        cars_root = os.path.join(data_dir, 'stanford_cars')
        train_ds = torchvision.datasets.ImageFolder(os.path.join(cars_root, 'train'), transform=train_tf)
        val_ds   = torchvision.datasets.ImageFolder(os.path.join(cars_root, 'test'),  transform=val_tf)

    elif dataset == 'pets':
        train_tf = _train_tf_large(img_size, mean, std)
        val_tf   = _val_tf_large(img_size, mean, std)
        train_ds = torchvision.datasets.OxfordIIITPet(data_dir, split='trainval', transform=train_tf, download=True)
        val_ds   = torchvision.datasets.OxfordIIITPet(data_dir, split='test',     transform=val_tf,   download=True)

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_CONFIGS)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ============================================================
# Training helpers
# ============================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.8):
    model.train()
    total_loss = 0.
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Mixup augmentation
        if mixup_alpha > 0 and model.training:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            idx = torch.randperm(imgs.size(0), device=device)
            imgs = lam * imgs + (1 - lam) * imgs[idx]
            labels_a, labels_b = labels, labels[idx]

            logits = model(imgs)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        pred = logits.argmax(dim=1)
        if mixup_alpha > 0:
            # For mixup, report accuracy against original labels
            correct += (pred == labels_a).sum().item()
        else:
            correct += (pred == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, 100.0 * correct / total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str,   default='cifar100', choices=list(DATASET_CONFIGS))
    parser.add_argument('--pe_type',    type=str,   default='ape', choices=MODEL_TYPES)
    parser.add_argument('--epochs',     type=int,   default=300)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--min_lr',     type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--drop_path',  type=float, default=0.1)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.8)
    parser.add_argument('--gpu',        type=int,   default=0)
    parser.add_argument('--num_workers', type=int,  default=8)
    parser.add_argument('--data_dir',   type=str,   default='./data')
    parser.add_argument('--results_dir', type=str,  default='./results')
    parser.add_argument('--save_freq',  type=int,   default=50)
    args = parser.parse_args()

    ds_cfg = DATASET_CONFIGS[args.dataset]

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = os.path.join(args.results_dir, args.dataset, args.pe_type)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"=== LPB-V Experiment: dataset={args.dataset}  pe_type={args.pe_type} ===")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader = get_loaders(args.dataset, args.data_dir, args.batch_size, args.num_workers)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Model
    model = build_model(
        pe_type=args.pe_type,
        image_size=ds_cfg['image_size'],
        patch_size=ds_cfg['patch_size'],
        num_classes=ds_cfg['num_classes'],
        drop_path_rate=args.drop_path,
    ).to(device)
    print(f"Parameters: {count_params(model)/1e6:.3f}M")

    # Optimizer
    # Separate weight decay: no decay on bias/norm params.
    # For ViT, also exclude learnable PE params from weight decay.
    is_resnet = args.pe_type in BASELINE_MODELS
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        vit_no_decay = not is_resnet and (
            'pos_embed' in name or 'r1_raw' in name or 'r2_raw' in name or
            'phi_star' in name or 's_raw' in name or 'slopes' in name
        )
        if p.ndim <= 1 or name.endswith('.bias') or vit_no_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = optim.AdamW([
        {'params': decay_params,    'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.999))

    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.lr, args.min_lr
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    log_path = os.path.join(run_dir, 'log.log')
    with open(log_path, 'w') as f:
        f.write('epoch lr train_loss train_acc val_loss val_acc\n')

    best_acc = 0.0
    for epoch in range(args.epochs):
        lr = scheduler.step(epoch)
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.mixup_alpha
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, os.path.join(run_dir, 'best.pth'))

        print(f"[{epoch+1:3d}/{args.epochs}] lr={lr:.2e}  "
              f"train: loss={train_loss:.4f} acc={train_acc:.2f}%  "
              f"val: loss={val_loss:.4f} acc={val_acc:.2f}%  "
              f"best={best_acc:.2f}%  {elapsed:.1f}s"
              + (" *" if is_best else ""))

        with open(log_path, 'a') as f:
            f.write(f'{epoch+1} {lr:.6f} {train_loss:.6f} {train_acc:.4f} {val_loss:.6f} {val_acc:.4f}\n')

        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, os.path.join(run_dir, f'ckpt_ep{epoch+1}.pth'))

    print(f"\n=== DONE: dataset={args.dataset}  pe_type={args.pe_type}  best_val_acc={best_acc:.2f}% ===")

    # Write final summary
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump({'dataset': args.dataset, 'pe_type': args.pe_type,
                   'best_val_acc': best_acc, 'epochs': args.epochs}, f, indent=2)

    return best_acc


if __name__ == '__main__':
    main()
