"""
Angular-term superposition — cleaner rebuild.

Layout:
  Row 0  (3 polar)   : model totals — positive petal filled, negative shown dashed
  Row 1  (3 cartesian): stacked-area per model, each harmonic in its own colour
  Row 2  (1 wide)    : all 3 model totals overlaid for direct comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── colour palette: one per harmonic order ────────────────────────────────────
C1 = '#4C72B0'   # blue   — cos(φ)
C2 = '#DD8452'   # orange — cos(2φ)
C3 = '#55A868'   # green  — cos(3φ)
COLORS = [C1, C2, C3]

phi  = np.linspace(0, 2 * np.pi, 1000)
deg  = np.degrees(phi)

h1   = np.cos(phi)
h2   = np.cos(2 * phi)
h3   = np.cos(3 * phi)

models = {
    'dlpb_aniso': {
        'harmonics':  [h1],
        'colors':     [C1],
        'labels':     [r'$\cos(\varphi)$'],
        'title':      'dlpb_aniso\n' + r'$= \cos(\varphi)$',
        'line_color': C1,
    },
    'dlpb_vm': {
        'harmonics':  [h1, h2],
        'colors':     [C1, C2],
        'labels':     [r'$\cos(\varphi)$', r'$+\cos(2\varphi)$'],
        'title':      'dlpb_vm\n' + r'$= \cos(\varphi) + \cos(2\varphi)$',
        'line_color': C2,
    },
    'dlpb_vm3': {
        'harmonics':  [h1, h2, h3],
        'colors':     [C1, C2, C3],
        'labels':     [r'$\cos(\varphi)$', r'$+\cos(2\varphi)$', r'$+\cos(3\varphi)$'],
        'title':      'dlpb_vm3\n' + r'$= \cos(\varphi)+\cos(2\varphi)+\cos(3\varphi)$',
        'line_color': C3,
    },
}


# ── polar plot: positive petal filled, negative part shown as dashed outline ──
def draw_polar_total(ax, harmonics, colors, title):
    total = sum(harmonics)

    # fill positive region
    pos = np.where(total >= 0, total, 0)
    neg_mag = np.where(total < 0, -total, 0)

    ax.fill_between(phi, 0, pos,     color=colors[-1], alpha=0.55, lw=0)
    ax.fill_between(phi, 0, neg_mag, color=colors[-1], alpha=0.20, lw=0,
                    label='_nolegend_')
    # boundary line of total (offset to keep non-negative for display)
    offset = max(0, -total.min()) + 0.02
    ax.plot(phi, total + offset, color='black', lw=1.8)

    # stacking breakdown: show each harmonic's positive ring
    cumulative = np.zeros_like(phi)
    for h, c in zip(harmonics, colors):
        prev_pos = np.maximum(0, cumulative)
        cumulative += h
        curr_pos = np.maximum(0, cumulative)
        # thin coloured boundary between layers
        ax.plot(phi, curr_pos, color=c, lw=0.9, alpha=0.6, linestyle='--')

    # label regions
    ax.set_yticklabels([])
    ax.tick_params(labelleft=False)
    ax.grid(True, alpha=0.2)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=14)


# ── Cartesian: components as individual fills + total as thick line ───────────
def draw_stacked_cart(ax, harmonics, colors, labels, title, ylim=(-3.3, 3.3)):
    # Each harmonic filled independently from 0 (so negative regions show below)
    for h, c, lbl in zip(harmonics, colors, labels):
        ax.fill_between(deg, 0, h, color=c, alpha=0.30, lw=0)
        ax.plot(deg, h, color=c, lw=1.3, linestyle='--', alpha=0.80, label=lbl)

    total = sum(harmonics)
    ax.fill_between(deg, 0, total, color='black', alpha=0.08, lw=0)
    ax.plot(deg, total, color='black', lw=2.5, label='total (sum)', zorder=5)
    ax.axhline(0, color='k', lw=0.8, alpha=0.45)
    ax.set_xlim(0, 360)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.88)
    ax.grid(True, alpha=0.22)


# ════════════════════════════════════════════════════════════════════════════
# Figure
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 13))
fig.suptitle(
    "Angular Term Superposition:  dlpb_aniso  /  dlpb_vm  /  dlpb_vm3",
    fontsize=14, fontweight='bold', y=0.995,
)

gs = GridSpec(3, 3, figure=fig,
              top=0.95, bottom=0.06,
              hspace=0.58, wspace=0.35,
              height_ratios=[1.15, 1.0, 0.85])

model_keys = ['dlpb_aniso', 'dlpb_vm', 'dlpb_vm3']

# ── Row 0: Polar model totals ─────────────────────────────────────────────────
for col, key in enumerate(model_keys):
    m   = models[key]
    ax  = fig.add_subplot(gs[0, col], projection='polar')
    draw_polar_total(ax, m['harmonics'], m['colors'], m['title'])

    # harmonic legend patches
    patches = [mpatches.Patch(color=c, alpha=0.7, label=lbl)
               for c, lbl in zip(m['colors'], m['labels'])]
    ax.legend(handles=patches, fontsize=7.5,
              loc='lower right', bbox_to_anchor=(1.42, -0.12),
              framealpha=0.85)

    if col == 0:
        ax.set_ylabel("Polar pattern\n(positive = filled, negative = light)",
                      labelpad=60, fontsize=8)

# ── Row 1: Cartesian stacked ──────────────────────────────────────────────────
for col, key in enumerate(model_keys):
    m  = models[key]
    ax = fig.add_subplot(gs[1, col])
    draw_stacked_cart(ax, m['harmonics'], m['colors'], m['labels'], m['title'])
    ax.set_xlabel("φ", fontsize=9)
    if col == 0:
        ax.set_ylabel("Angular bias value", fontsize=9)

# ── Row 2: All models overlaid ────────────────────────────────────────────────
ax_cmp = fig.add_subplot(gs[2, :])

totals   = [sum(models[k]['harmonics']) for k in model_keys]
lc       = [C1, C2, C3]
ls       = ['-', '--', '-.']

for total, c, ls_, key in zip(totals, lc, ls, model_keys):
    ax_cmp.plot(deg, total, color=c, lw=2.2, linestyle=ls_, label=key)

# shade the "added" regions between consecutive models
ax_cmp.fill_between(deg, totals[0], totals[1],
                    alpha=0.15, color=C2,
                    label=r'diff: vm − aniso  ($+\cos 2\varphi$)')
ax_cmp.fill_between(deg, totals[1], totals[2],
                    alpha=0.15, color=C3,
                    label=r'diff: vm3 − vm  ($+\cos 3\varphi$)')

ax_cmp.axhline(0, color='k', lw=0.7, alpha=0.4)
ax_cmp.set_xlim(0, 360)
ax_cmp.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax_cmp.set_xticklabels(['0°','45°','90°','135°','180°','225°','270°','315°','360°'])
ax_cmp.set_title("Direct comparison of model totals  "
                 "(shading = difference between consecutive models)",
                 fontsize=10)
ax_cmp.set_xlabel("φ  (direction from query patch to key patch)", fontsize=9)
ax_cmp.set_ylabel("Angular bias", fontsize=9)
ax_cmp.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax_cmp.grid(True, alpha=0.22)

out = "/data1/LPB-V/plot/angular_superposition.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
