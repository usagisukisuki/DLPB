"""Summarize experiment results from all PE variants."""

import os
import json
import csv
import sys
import argparse

def get_available_datasets(base_dir='./results'):
    if not os.path.isdir(base_dir):
        return []
    return sorted(d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)))

parser = argparse.ArgumentParser(description='Summarize experiment results')
available = get_available_datasets()
parser.add_argument(
    'dataset',
    nargs='?',
    choices=available if available else None,
    default=available[0] if len(available) == 1 else None,
    help=f'Dataset to summarize. Available: {available}',
)
args_parsed = parser.parse_args()

if args_parsed.dataset is None:
    if not available:
        print('No datasets found in ./results/')
        sys.exit(1)
    print(f'Available datasets: {available}')
    print(f'Usage: python summarize_results.py <dataset>')
    sys.exit(0)

DATASET = args_parsed.dataset
RESULTS_DIR = os.path.join('./results', DATASET)
print(f'Dataset: {DATASET}')

# Baselines + the proposed methods only.
# Other variants we explored remain registered in models.py but are kept
# out of this summary table.
PE_TYPES = [
    # baselines
    'resnet18', 'no_pe', 'ape', 'alibi_2d', 'rpb', 'cpb', 'rope_2d', 'kerple_log_2d',
    # proposed
    'dlpb', 'dlpb_O2', 'dlpb_O3',
    'dlpb_rope_2d', 'dlpb_O2_rope_2d', 'dlpb_O3_rope_2d',
]

# ── Accuracy table ────────────────────────────────────────────
print(f"\n{'PE Type':15s}  {'Best Val Acc':>12s}  {'Status':>10s}")
print("-" * 45)

rows = []
for pe in PE_TYPES:
    result_path = os.path.join(RESULTS_DIR, pe, 'result.json')
    log_path    = os.path.join(RESULTS_DIR, pe, 'log.csv')

    if os.path.exists(result_path):
        with open(result_path) as f:
            d = json.load(f)
        acc = d.get('best_val_acc', 0.)
        print(f"{pe:15s}  {acc:>12.2f}%  {'done':>10s}")
        rows.append((pe, acc))
    elif os.path.exists(log_path):
        with open(log_path) as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
        if last:
            print(f"{pe:15s}  {float(last['val_acc']):>12.2f}%  {'ep'+last['epoch']:>10s}")
        else:
            print(f"{pe:15s}  {'—':>12s}  {'running':>10s}")
    else:
        print(f"{pe:15s}  {'—':>12s}  {'pending':>10s}")

if rows:
    print()
    best_pe, best_acc = max(rows, key=lambda x: x[1])
    print(f"Best: {best_pe} @ {best_acc:.2f}%")

# ── Parameter count table ─────────────────────────────────────
try:
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from models import build_model, count_params, count_pe_params, PE_TYPES as MODEL_PE_TYPES

    print(f"\n{'PE Type':15s}  {'Total':>10s}  {'PE only':>10s}  {'PE share':>10s}")
    print("-" * 52)
    with torch.no_grad():
        for pe in PE_TYPES:
            if pe not in MODEL_PE_TYPES:
                continue
            m = build_model(pe)
            total = count_params(m)
            pe_n  = count_pe_params(m)
            print(f"{pe:15s}  {total/1e6:>8.3f}M  {pe_n/1e3:>8.3f}K  {100*pe_n/total:>8.3f}%")
except Exception as e:
    print(f"\n[param table skipped: {e}]")
