import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

datasets = {
    "Stanford Cars": "cars.json",
    "CIFAR-100": "cifar100.json",
    "Flowers-102": "flowers102.json",
    "Oxford Pets": "pets.json",
    "ImageNet-100": "imagenet100.json",
}

results_dir = Path("results/resolution")

# Method display names and style grouping
method_styles = {
    "no_pe":          {"label": "No PE",           "ls": "--",  "lw": 1.2, "marker": "o", "group": "baseline"},
    "ape":            {"label": "APE",              "ls": "--",  "lw": 1.2, "marker": "s", "group": "baseline"},
    "alibi_2d":       {"label": "ALiBi-2D",         "ls": "--",  "lw": 1.2, "marker": "^", "group": "baseline"},
    "rope_2d":         {"label": "RoPE-2D",          "ls": "--",  "lw": 1.2, "marker": "D", "group": "baseline"},
    #"rpb":            {"label": "RPB",              "ls": "--",  "lw": 1.2, "marker": "v", "group": "baseline"},
    "cpb":            {"label": "CPB",              "ls": "-.",  "lw": 1.5, "marker": "P", "group": "baseline"},
    #"kerple_log_2d":       {"label": "KEPLAR",      "ls": "-",   "lw": 1.5, "marker": "o", "group": "baseline"},
    "dlpb":  {"label": "DLPB", "ls": "-",   "lw": 2.0, "marker": "s", "group": "dlpb"},
    #"dlpb_vm_sc_fix":     {"label": "LPB-V (vm+sc)",    "ls": "-",   "lw": 2.0, "marker": "^", "group": "dlpb"},
    "dlpb_O3":    {"label": "DLPB-O3",   "ls": "-",   "lw": 2.5, "marker": "D", "group": "dlpb"},
    "dlpb_rope_2d":  {"label": "DLPB+RoPE-2D", "ls": "-",   "lw": 2.0, "marker": "s", "group": "dlpb_rope_2d"},
    #"dlpb_vm_sc_fix_rope_2d":  {"label": "LPB-V (vm+sc+rope)",  "ls": ":", "lw": 2.0, "marker": "^", "group": "dlpb_rope_2d"},
    "dlpb_O3_rope_2d": {"label": "DLPB-O3+RoPE-2D", "ls": ":", "lw": 2.0, "marker": "D", "group": "dlpb_rope_2d"},
}

# Assign a distinct color to each method
_distinct_colors = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
]
method_colors = {
    m: _distinct_colors[i % len(_distinct_colors)]
    for i, m in enumerate(method_styles)
}

for title, fname in datasets.items():
    json_path = results_dir / fname
    if not json_path.exists():
        print(f"Skip (not found): {json_path}")
        continue

    with open(json_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))

    for method, st in method_styles.items():
        if method not in data:
            continue
        vals = data[method]
        xs = sorted(int(k) for k, v in vals.items() if v is not None)
        ys = [vals[str(x)] for x in xs]

        ax.plot(
            xs, ys,
            label=st["label"],
            linestyle=st["ls"],
            linewidth=st["lw"],
            marker=st["marker"],
            markersize=4,
            color=method_colors[method],
        )

    ax.set_title(f"Resolution Generalization — {title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Resolution" + (" (patch size)" if "CIFAR" in title else " (px)"), fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.18),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    stem = fname.replace(".json", "")
    out_path = f"plot/resolution_plot_{stem}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
