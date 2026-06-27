"""
Local figure generation from wandb-downloaded artifacts.
No GPU, no RSNA dataset, no training required.

Run from: Anomaly-detection-project/
  python local_figures.py

Produces: local_report/ with all figures + CSV summary
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE     = os.path.dirname(os.path.abspath(__file__))
SRC_JSON = os.path.join(BASE, 'all_results', 'results')
OUT_DIR  = os.path.join(BASE, 'local_report')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────
with open(f'{SRC_JSON}/results.json') as f:
    all_results = json.load(f)
with open(f'{SRC_JSON}/loss_history.json') as f:
    loss_history = json.load(f)

CONDITIONS = ['C1','C2','C3','C4','C5','C6','C7']
LABELS = {
    'C1': 'CNN-AE',
    'C2': 'VAE',
    'C3': 'CNN-AAE',
    'C4': 'CNN-RE-Attn-AAE\n(Ours)',
    'C5': 'ResNet Frozen',
    'C6': 'ResNet Partial FT',
    'C7': 'ResNet Mostly FT',
}
COLORS = {
    'C1': '#9e9e9e', 'C2': '#78909c', 'C3': '#42a5f5',
    'C4': '#e53935', 'C5': '#8d6e63', 'C6': '#43a047', 'C7': '#fb8c00',
}

auc_roc = {c: all_results[c]['auc_roc'] for c in CONDITIONS}
auc_pr  = {c: all_results[c]['auc_pr']  for c in CONDITIONS}
f1      = {c: all_results[c]['f1']      for c in CONDITIONS}

# ── Figure 1: Ablation bar chart (AUC-ROC) ──────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(CONDITIONS))
bars = ax.bar(x, [auc_roc[c] for c in CONDITIONS],
              color=[COLORS[c] for c in CONDITIONS],
              edgecolor='white', linewidth=1.2, width=0.6)
ax.axhline(0.5, color='black', lw=0.8, ls='--', alpha=0.4, label='Random baseline')
for bar, cond in zip(bars, CONDITIONS):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
            f'{h:.4f}', ha='center', va='bottom', fontsize=9,
            fontweight='bold' if cond == 'C4' else 'normal')
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=9)
ax.set_ylabel('AUC-ROC', fontsize=11)
ax.set_title('Ablation Study — AUC-ROC by Condition\n'
             'C4 (red) = full novel method: CNN + RE-Attention + AAE', fontsize=12)
ax.set_ylim(0.4, 0.92)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_ablation_auroc.png', dpi=150)
plt.close()
print("fig1_ablation_auroc.png")

# ── Figure 2: Three-metric grouped bar chart ─────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(CONDITIONS))
w = 0.26
b1 = ax.bar(x - w, [auc_roc[c] for c in CONDITIONS], w, label='AUC-ROC',
            color=[COLORS[c] for c in CONDITIONS], alpha=0.95)
b2 = ax.bar(x,     [auc_pr[c]  for c in CONDITIONS], w, label='AUC-PR',
            color=[COLORS[c] for c in CONDITIONS], alpha=0.65)
b3 = ax.bar(x + w, [f1[c]      for c in CONDITIONS], w, label='F1',
            color=[COLORS[c] for c in CONDITIONS], alpha=0.40)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in CONDITIONS], fontsize=9)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Ablation — AUC-ROC / AUC-PR / F1 per Condition', fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend(['AUC-ROC (dark)', 'AUC-PR (mid)', 'F1 (light)'], fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_three_metrics.png', dpi=150)
plt.close()
print("fig2_three_metrics.png")

# ── Figure 3: Loss curves ────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for idx, cond in enumerate(CONDITIONS):
    ax = axes[idx]
    hist = loss_history.get(cond, [])
    if hist:
        ax.plot(hist, color=COLORS[cond], lw=1.8)
        ax.set_title(f'{cond} — {LABELS[cond].replace(chr(10)," ")}', fontsize=9)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Recon Loss', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='grey')
        ax.set_title(cond, fontsize=9)
axes[-1].set_visible(False)
fig.suptitle('Training Loss Convergence — All 7 Conditions', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3_loss_curves.png")

# ── Figure 4: Fusion vs SSIM gain ────────────────────────────────────
fuse_conds = ['C3','C4','C5','C6','C7']
ssim_vals  = [all_results[c]['auc_roc'] for c in fuse_conds]
fuse_vals  = [all_results.get(f'{c}_fuse', {}).get('auc_roc', 0) for c in fuse_conds]
x = np.arange(len(fuse_conds))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, ssim_vals, w, label='SSIM score only', color='#90caf9', edgecolor='white')
ax.bar(x + w/2, fuse_vals, w, label='Fusion (SSIM + disc)', color='#ef9a9a', edgecolor='white')
for i, (s, f) in enumerate(zip(ssim_vals, fuse_vals)):
    delta = f - s
    ax.text(i, max(s, f) + 0.006, f'{delta:+.3f}', ha='center', fontsize=9,
            color='green' if delta > 0 else 'red')
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n{LABELS[c].split(chr(10))[0]}' for c in fuse_conds], fontsize=9)
ax.set_ylabel('AUC-ROC', fontsize=11)
ax.set_title('Score Fusion Effect: SSIM-only vs SSIM + Discriminator', fontsize=12)
ax.set_ylim(0.4, 0.92)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_fusion_gain.png', dpi=150)
plt.close()
print("fig4_fusion_gain.png")

# ── Figure 5: Ablation chain (C1→C4) ─────────────────────────────────
chain = ['C1','C3','C4']
chain_labels = ['C1\nCNN-AE\n(baseline)', 'C3\nCNN-AAE\n(+adversarial)', 'C4\nCNN-RE-Attn-AAE\n(+attention, ours)']
vals  = [auc_roc[c] for c in chain]
deltas = [0] + [vals[i] - vals[i-1] for i in range(1, len(vals))]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(range(len(chain)), vals,
              color=['#9e9e9e','#42a5f5','#e53935'],
              edgecolor='white', linewidth=1.2, width=0.5)
for i, (bar, d) in enumerate(zip(bars, deltas)):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
            f'{h:.4f}', ha='center', fontsize=11, fontweight='bold')
    if i > 0:
        ax.annotate(f'{d:+.4f}', xy=(i, h/2), ha='center', fontsize=10,
                    color='white', fontweight='bold')
# Draw arrows between bars
for i in range(len(chain)-1):
    ax.annotate('', xy=(i+1-0.28, vals[i+1]),
                xytext=(i+0.28, vals[i]),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.set_xticks(range(len(chain)))
ax.set_xticklabels(chain_labels, fontsize=10)
ax.set_ylabel('AUC-ROC', fontsize=11)
ax.set_ylim(0.55, 0.90)
ax.set_title('CNN Ablation Chain: Each Component\'s Contribution', fontsize=12)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_ablation_chain.png', dpi=150)
plt.close()
print("fig5_ablation_chain.png")

# ── Figure 6: ResNet comparison (C4 vs C5 vs C6 vs C7) ───────────────
resnet_conds  = ['C4','C5','C6','C7']
resnet_labels = ['C4\nCNN enc\n(scratch)',
                 'C5\nResNet\nfull freeze',
                 'C6\nResNet\npartial FT',
                 'C7\nResNet\nmostly FT']
resnet_vals   = [auc_roc[c] for c in resnet_conds]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(4), resnet_vals,
              color=[COLORS[c] for c in resnet_conds],
              edgecolor='white', linewidth=1.2, width=0.55)
for bar, v, c in zip(bars, resnet_vals, resnet_conds):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
            f'{v:.4f}', ha='center', fontsize=11,
            fontweight='bold' if c in ['C4','C6'] else 'normal')
ax.axhline(auc_roc['C4'], color=COLORS['C4'], lw=1.2, ls='--', alpha=0.5,
           label=f'C4 scratch CNN ({auc_roc["C4"]:.4f})')
ax.set_xticks(range(4))
ax.set_xticklabels(resnet_labels, fontsize=10)
ax.set_ylabel('AUC-ROC', fontsize=11)
ax.set_title('Transfer Learning Comparison: Scratch CNN vs ResNet Variants\n'
             'C5 collapse shows full freeze fails; C6 partial FT recovers', fontsize=11)
ax.set_ylim(0.50, 0.90)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_resnet_comparison.png', dpi=150)
plt.close()
print("fig6_resnet_comparison.png")

# ── Figure 7: Summary table as figure ────────────────────────────────
import matplotlib.table as mtable
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
col_labels = ['Condition', 'Architecture', 'AUC-ROC', 'AUC-PR', 'F1',
              'Fusion AUC', 'vs C1', 'Disc stable']
rows = []
disc_stable = {'C1':'-','C2':'-','C3':'yes','C4':'yes',
               'C5':'COLLAPSED','C6':'yes','C7':'yes'}
arch = {
    'C1':'CNN-AE','C2':'VAE','C3':'CNN-AAE',
    'C4':'CNN-RE-Attn-AAE (Ours)','C5':'ResNet frozen',
    'C6':'ResNet partial FT','C7':'ResNet mostly FT'
}
for c in CONDITIONS:
    r = all_results[c]
    fuse_auc = all_results.get(f'{c}_fuse', {}).get('auc_roc', float('nan'))
    rows.append([
        c, arch[c],
        f"{r['auc_roc']:.4f}", f"{r['auc_pr']:.4f}", f"{r['f1']:.4f}",
        f"{fuse_auc:.4f}" if not np.isnan(fuse_auc) else '-',
        f"{r['auc_roc'] - auc_roc['C1']:+.4f}",
        disc_stable[c],
    ])
tbl = ax.table(cellText=rows, colLabels=col_labels,
               cellLoc='center', loc='center',
               bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
# Highlight C4 row
for j in range(len(col_labels)):
    tbl[4, j].set_facecolor('#fde8e8')
    tbl[4, j].set_text_props(fontweight='bold')
# Header style
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#263238')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Ablation Results Summary', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_results_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig7_results_table.png")

# ── CSV export ────────────────────────────────────────────────────────
import csv
with open(f'{OUT_DIR}/ablation_results.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['condition','label','auc_roc','auc_pr','f1',
                'auc_roc_disc','auc_roc_fusion','delta_vs_c1','disc_stable'])
    for c in CONDITIONS:
        r = all_results[c]
        w.writerow([
            c, arch[c],
            round(r['auc_roc'],4), round(r['auc_pr'],4), round(r['f1'],4),
            round(all_results.get(f'{c}_disc',{}).get('auc_roc', float('nan')),4),
            round(all_results.get(f'{c}_fuse',{}).get('auc_roc', float('nan')),4),
            round(r['auc_roc'] - auc_roc['C1'], 4),
            disc_stable[c],
        ])
print("ablation_results.csv")

print(f"\nAll outputs in: {OUT_DIR}/")
