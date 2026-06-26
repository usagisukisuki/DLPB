"""
Polar-only outputs (no Cartesian panels):
  1. angular_harmonics_comparison.png  — cos φ / cos 2φ / cos 3φ side-by-side
  2. angular_model_aniso.png
  3. angular_model_vm.png
  4. angular_model_vm3.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

C1, C2, C3 = '#4C72B0', '#DD8452', '#55A868'

phi = np.linspace(0, 2 * np.pi, 2000)

h1 = np.cos(phi)
h2 = np.cos(2 * phi)
h3 = np.cos(3 * phi)


def draw_petal(ax, values, color, alpha_pos=0.55, alpha_neg=0.20, lw=1.6):
    """Positive region filled, negative region lighter at opposite side."""
    pos = np.where(values >= 0, values, 0)
    neg = np.where(values < 0, -values, 0)
    ax.fill_between(phi, 0, pos, color=color, alpha=alpha_pos, lw=0)
    ax.fill_between((phi + np.pi) % (2 * np.pi), 0, neg,
                    color=color, alpha=alpha_neg, lw=0)
    ax.plot(phi, pos, color=color, lw=lw)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.2)


def add_peak_arrows(ax, peak_degs, color, r_tip=0.95, r_label=1.22):
    for d in peak_degs:
        r = np.radians(d)
        ax.annotate('', xy=(r, r_tip), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=2.0, shrinkA=0, shrinkB=0))
        ax.text(r, r_label, f'{d}°',
                ha='center', va='center', fontsize=9,
                color=color, fontweight='bold')


def add_period_arc(ax, period_deg, color, r_arc=0.68):
    arc = np.linspace(0, np.radians(period_deg), 200)
    ax.plot(arc, np.full_like(arc, r_arc), color=color, lw=3, alpha=0.75)
    ax.text(np.radians(period_deg / 2), r_arc + 0.16,
            f'period {period_deg}°',
            ha='center', va='center', fontsize=8.5,
            color=color, fontweight='bold')


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: cos φ / cos 2φ / cos 3φ comparison (3 polar plots)
# ════════════════════════════════════════════════════════════════════════════
harmonic_specs = [
    dict(h=h1, color=C1, period=360, peaks=[0],
         title=r'$\cos\,\theta$' + '\n(1st order · 360° period)'),
    dict(h=h2, color=C2, period=180, peaks=[0, 180],
         title=r'$\cos\,2\theta$' + '\n(2nd order · 180° period)'),
    dict(h=h3, color=C3, period=120, peaks=[0, 120, 240],
         title=r'$\cos\,3\theta$' + '\n(3rd order · 120° period)'),
]

fig1, axes = plt.subplots(1, 3, figsize=(14, 5.5),
                           subplot_kw={'projection': 'polar'})

for ax, spec in zip(axes, harmonic_specs):
    draw_petal(ax, spec['h'], spec['color'])
    add_peak_arrows(ax, spec['peaks'], spec['color'])
    add_period_arc(ax, spec['period'], spec['color'])
    ax.set_title(spec['title'], fontsize=11, fontweight='bold',
                 pad=18, color=spec['color'])

plt.tight_layout()
out1 = '/data1/LPB-V/plot/angular_harmonics_comparison.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close(fig1)


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: all 3 models side-by-side in one PNG
# ════════════════════════════════════════════════════════════════════════════
model_specs = [
    dict(
        harmonics=[h1],
        colors=[C1],
        h_labels=[r'$\cos\,\theta$'],
        title='DLPB',
        equation=r'$\cos\,\theta$',
        peaks={1: [0]},
    ),
    dict(
        harmonics=[h1, h2],
        colors=[C1, C2],
        h_labels=[r'$\cos\,\theta$', r'$\cos\,2\theta$'],
        title='DLPB-O2',
        equation=r'$\cos\,\theta + \cos\,2\theta$',
        peaks={1: [0], 2: [0, 180]},
    ),
    dict(
        harmonics=[h1, h2, h3],
        colors=[C1, C2, C3],
        h_labels=[r'$\cos\,\theta$', r'$\cos\,2\theta$', r'$\cos\,3\theta$'],
        title='DLPB-O3',
        equation=r'$\cos\,\theta + \cos\,2\theta + \cos\,3\theta$',
        peaks={1: [0], 2: [0, 180], 3: [0, 120, 240]},
    ),
]

r_tips   = [0.70, 0.82, 0.95]
r_labels = [0.88, 1.02, 1.18]

fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4.2),
                            subplot_kw={'projection': 'polar'})
fig2.subplots_adjust(left=0.02, right=0.88, top=0.82, bottom=0.05,
                     wspace=0.45)

for ax, spec in zip(axes2, model_specs):
    total = sum(spec['harmonics'])

    for h, c in zip(spec['harmonics'], spec['colors']):
        ax.plot(phi, np.where(h >= 0, h, 0),
                color=c, lw=1.2, linestyle='--', alpha=0.65)

    draw_petal(ax, total, spec['colors'][-1],
               alpha_pos=0.45, alpha_neg=0.15, lw=2.2)

    for k, (c, pks) in enumerate(
            zip(spec['colors'], spec['peaks'].values()), start=0):
        add_peak_arrows(ax, pks, c,
                        r_tip=r_tips[k], r_label=r_labels[k])

    ax.set_yticklabels([])
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{spec['title']}\n{spec['equation']}",
                 fontsize=10, fontweight='bold', pad=14)

    patches = [mpatches.Patch(color=c, alpha=0.7, label=lbl)
               for c, lbl in zip(spec['colors'], spec['h_labels'])]
    ax.legend(handles=patches, fontsize=8,
              loc='lower right', bbox_to_anchor=(1.42, -0.08),
              framealpha=0.9)

out2 = '/data1/LPB-V/plot/angular_models_comparison.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight', pad_inches=0.1)
print(f'Saved: {out2}')
plt.close(fig2)
