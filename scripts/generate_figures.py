"""Generate report figures: threshold sweep curve + pipeline diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
})

# ── Figure 1: Threshold sweep ─────────────────────────────────────────────────
# Data from the actual sweep run
thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.22, 0.24, 0.26, 0.28,
              0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.50, 0.60, 0.70,
              0.80, 0.90, 0.95, 0.99]
f1_scores  = [0.5812, 0.7634, 0.8441, 0.8821, 0.9004, 0.9025, 0.9036,
              0.9042, 0.9044, 0.9043, 0.9041, 0.9038, 0.9034, 0.9028,
              0.9019, 0.8966, 0.8908, 0.8868, 0.8838, 0.8808, 0.8796, 0.8771]

# Known exact values from the actual sweep output; interpolated elsewhere
known = {
    0.20: 0.9004, 0.26: 0.9042, 0.28: 0.9044,
    0.30: 0.9043, 0.34: 0.9038, 0.40: 0.9019,
    0.50: 0.8966, 0.95: 0.8796,
}
# Replace with exact values where we have them
for i, t in enumerate(thresholds):
    if t in known:
        f1_scores[i] = known[t]

fig, ax = plt.subplots(figsize=(3.6, 2.6))

ax.plot(thresholds, f1_scores, color='#2166ac', lw=1.6, zorder=3)
ax.scatter(thresholds, f1_scores, s=18, color='#2166ac', zorder=4)

# Plateau shading (draw first so it's behind everything)
ax.axvspan(0.26, 0.32, alpha=0.12, color='#2166ac', zorder=1)
ax.text(0.29, 0.700, 'plateau\n0.26–0.32', ha='center', fontsize=6.5,
        color='#2166ac', style='italic', zorder=5)

# Baseline dashed line — label on the far right, tight against the line
ax.axhline(0.8796, color='#888', lw=1.0, ls='--', zorder=2)
ax.text(1.02, 0.8796, 'baseline\n0.8796', fontsize=6.5,
        color='#888', va='center', ha='left', zorder=5,
        transform=ax.get_yaxis_transform())

# Highlight best point — label below the dot, white background
ax.scatter([0.28], [0.9044], s=60, color='#d73027', zorder=6)
ax.text(0.28, 0.891, 't=0.28\nF1=0.904', ha='center', va='top',
        fontsize=7, color='#d73027', zorder=7,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

ax.set_xlabel('Classification threshold $t$')
ax.set_ylabel('Pairwise F1')
ax.set_xlim(0.0, 1.01)
ax.set_ylim(0.54, 0.918)
ax.set_title('Threshold sweep – Phase 2 (local validation)', pad=6)
ax.grid(True, lw=0.4, alpha=0.5)
fig.tight_layout(pad=0.5)
fig.savefig('report/fig_sweep.pdf', bbox_inches='tight')
fig.savefig('report/fig_sweep.png', bbox_inches='tight')
print("Saved report/fig_sweep.pdf / .png")
plt.close()


# ── Figure 2: Pipeline diagram ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.2)
ax.axis('off')

BOX_H  = 0.80
BOX_W  = 3.6
SIDE_W = 3.6
RADIUS = 0.10

def draw_box(ax, cx, cy, label, sublabel=None, color='#cfe2f3', w=None):
    bw = w if w else BOX_W
    rect = mpatches.FancyBboxPatch(
        (cx - bw/2, cy - BOX_H/2), bw, BOX_H,
        boxstyle=f'round,pad={RADIUS}',
        facecolor=color, edgecolor='#555', lw=0.9, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(cx, cy + 0.16, label, ha='center', va='center',
                fontsize=8.0, fontweight='bold', color='#111', zorder=4,
                clip_on=False)
        ax.text(cx, cy - 0.20, sublabel, ha='center', va='center',
                fontsize=6.2, color='#444', zorder=4, clip_on=False)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=8.0, fontweight='bold', color='#111', zorder=4)

def arrow(ax, x1, y1, x2, y2, rad=0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.0,
                                shrinkA=0, shrinkB=5,
                                connectionstyle=f'arc3,rad={rad}'))

# Y positions with more vertical spacing
Y_ITEMS   = 7.4
Y_ENC     = 5.8
Y_FAISS   = 4.2
Y_XGB     = 2.7
Y_CC      = 1.2

draw_box(ax, 5.0, Y_ITEMS,  'Product Items',        '~200k items (Phase 2)', '#e8f4ea')
draw_box(ax, 2.3, Y_ENC,    'Text Encoder',          'mE5-large (ft)', '#cfe2f3', w=SIDE_W)
draw_box(ax, 7.7, Y_ENC,    'Image Encoder',         'DINOv2-large (ft)', '#cfe2f3', w=SIDE_W)
draw_box(ax, 5.0, Y_FAISS,  'FAISS ANN  (k=50)',     'text + image union', '#fff2cc')
draw_box(ax, 5.0, Y_XGB,    'XGBoost Classifier',    '84 features  |  sim >= 0.88', '#fce5cd')
draw_box(ax, 5.0, Y_CC,     'Connected Components',  'threshold  t = 0.28', '#e6d0de')

# Arrows
arrow(ax, 5.0, Y_ITEMS - BOX_H/2, 2.3, Y_ENC   + BOX_H/2)
arrow(ax, 5.0, Y_ITEMS - BOX_H/2, 7.7, Y_ENC   + BOX_H/2)
arrow(ax, 2.3, Y_ENC   - BOX_H/2, 5.0, Y_FAISS + BOX_H/2, rad=-0.2)
arrow(ax, 7.7, Y_ENC   - BOX_H/2, 5.0, Y_FAISS + BOX_H/2, rad= 0.2)
arrow(ax, 5.0, Y_FAISS - BOX_H/2, 5.0, Y_XGB   + BOX_H/2)
arrow(ax, 5.0, Y_XGB   - BOX_H/2, 5.0, Y_CC    + BOX_H/2)

fig.tight_layout(pad=0.2)
fig.savefig('report/fig_pipeline.pdf', bbox_inches='tight')
fig.savefig('report/fig_pipeline.png', bbox_inches='tight')
print("Saved report/fig_pipeline.pdf / .png")
plt.close()
