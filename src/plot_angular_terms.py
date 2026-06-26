"""
Comparison of angular terms for dlpb_aniso, dlpb_vm, and dlpb_vm3.

Each model's angular bias (distance term removed, phi_star=0):
  aniso : s * cos(phi)
  vm    : s1*cos(phi) + s2*cos(2*phi)
  vm3   : s1*cos(phi) + s2*cos(2*phi) + s3*cos(3*phi)

We show three views:
  1. Polar plots of the total angular term (per model)
  2. Cartesian plot of each harmonic component
  3. 2-D heatmap of the angular term on a pixel grid (distance fixed)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

phi = np.linspace(0, 2 * np.pi, 1000)

# softplus(-2.0) ≈ 0.127  (initial s value from models.py)
s_init = np.log1p(np.exp(-2.0))   # ≈ 0.127

# ---- individual harmonics (phi_star = 0, s = 1 for shape comparison) ----
h1 = np.cos(phi)          # 1st order (aniso)
h2 = np.cos(2 * phi)      # 2nd order (vm adds this)
h3 = np.cos(3 * phi)      # 3rd order (vm3 adds this)

# ---- model totals with equal weights s=1 (shows shape) ----
ang_aniso = h1
ang_vm    = h1 + h2
ang_vm3   = h1 + h2 + h3

# ---- model totals with init weights s=softplus(-2) ----
ang_aniso_w = s_init * h1
ang_vm_w    = s_init * (h1 + h2)
ang_vm3_w   = s_init * (h1 + h2 + h3)

# ---- 2-D grid for heatmap ----
G = 200
xv = np.linspace(-1, 1, G)
yv = np.linspace(-1, 1, G)
XX, YY = np.meshgrid(xv, yv)
PHI = np.arctan2(YY, XX)   # angle from query patch to key patch

heat_aniso = np.cos(PHI)
heat_vm    = np.cos(PHI) + np.cos(2 * PHI)
heat_vm3   = np.cos(PHI) + np.cos(2 * PHI) + np.cos(3 * PHI)

# ================================================================
# Figure layout
# ================================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Angular Term Comparison: dlpb_aniso / dlpb_vm / dlpb_vm3",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.55, wspace=0.35,
                       top=0.92, bottom=0.05)

COLORS = {'aniso': '#1f77b4', 'vm': '#ff7f0e', 'vm3': '#2ca02c'}
LABELS = {
    'aniso': r'dlpb_aniso:  $\cos(\varphi)$',
    'vm':    r'dlpb_vm:  $\cos(\varphi)+\cos(2\varphi)$',
    'vm3':   r'dlpb_vm3:  $\cos(\varphi)+\cos(2\varphi)+\cos(3\varphi)$',
}

# ----------------------------------------------------------------
# Row 0: Polar plots (total angular term, equal weights)
# ----------------------------------------------------------------
for col, (key, ang) in enumerate(
        [('aniso', ang_aniso), ('vm', ang_vm), ('vm3', ang_vm3)]):
    ax = fig.add_subplot(gs[0, col], projection='polar')
    # Shift so negative values don't fold; use r = ang - min
    offset = max(0, -ang.min()) + 0.05
    r = ang + offset
    ax.plot(phi, r, color=COLORS[key], linewidth=2)
    ax.fill(phi, r, alpha=0.15, color=COLORS[key])
    # mark max direction
    idx_max = np.argmax(ang)
    ax.annotate('', xy=(phi[idx_max], r[idx_max]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.set_title(LABELS[key], fontsize=9, pad=12)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.set_ylabel("Polar (equal weights)", labelpad=40, fontsize=9)

# ----------------------------------------------------------------
# Row 1: Cartesian — individual harmonics + model sums
# ----------------------------------------------------------------
# Left: individual harmonic components
ax_cart = fig.add_subplot(gs[1, :2])
deg = np.degrees(phi)
ax_cart.plot(deg, h1, '--', color='gray',   lw=1.5, label=r'$\cos(\varphi)$ [1st]')
ax_cart.plot(deg, h2, ':',  color='purple', lw=1.5, label=r'$\cos(2\varphi)$ [2nd]')
ax_cart.plot(deg, h3, '-.', color='brown',  lw=1.5, label=r'$\cos(3\varphi)$ [3rd]')
ax_cart.set_xlabel("φ (degrees)")
ax_cart.set_ylabel("Value")
ax_cart.set_title("Individual harmonic components", fontsize=10)
ax_cart.axhline(0, color='k', lw=0.5, alpha=0.4)
ax_cart.set_xticks([0, 60, 90, 120, 180, 240, 270, 300, 360])
ax_cart.legend(fontsize=9)
ax_cart.grid(True, alpha=0.3)

# Right: model sums
ax_sum = fig.add_subplot(gs[1, 2])
ax_sum.plot(deg, ang_aniso, color=COLORS['aniso'], lw=2, label='dlpb_aniso')
ax_sum.plot(deg, ang_vm,    color=COLORS['vm'],    lw=2, label='dlpb_vm')
ax_sum.plot(deg, ang_vm3,   color=COLORS['vm3'],   lw=2, label='dlpb_vm3')
ax_sum.axhline(0, color='k', lw=0.5, alpha=0.4)
ax_sum.set_xlabel("φ (degrees)")
ax_sum.set_title("Model angular sums (equal weights)", fontsize=10)
ax_sum.set_xticks([0, 90, 180, 270, 360])
ax_sum.legend(fontsize=8)
ax_sum.grid(True, alpha=0.3)

# ----------------------------------------------------------------
# Row 2: 2-D heatmaps of angular term on pixel grid
# ----------------------------------------------------------------
vmax = max(abs(heat_aniso).max(), abs(heat_vm).max(), abs(heat_vm3).max())
kw = dict(vmin=-vmax, vmax=vmax, cmap='RdBu_r', origin='lower',
          extent=[-1, 1, -1, 1])

for col, (key, heat, title) in enumerate([
        ('aniso', heat_aniso, 'dlpb_aniso\n' + r'$\cos(\varphi)$'),
        ('vm',    heat_vm,    'dlpb_vm\n'    + r'$\cos(\varphi)+\cos(2\varphi)$'),
        ('vm3',   heat_vm3,   'dlpb_vm3\n'   + r'$\cos(\varphi)+\cos(2\varphi)+\cos(3\varphi)$'),
]):
    ax = fig.add_subplot(gs[2, col])
    im = ax.imshow(heat, **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Δx (key relative to query)")
    if col == 0:
        ax.set_ylabel("Δy")
    ax.axhline(0, color='k', lw=0.3, alpha=0.5)
    ax.axvline(0, color='k', lw=0.3, alpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

out = "/data1/LPB-V/plot/angular_term_comparison.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
