"""
CausalRec-Bench — Generate Charts
Creates publication quality figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs('figures', exist_ok=True)

print("Loading results...")
results = pd.read_csv('results/final_results.csv')

model_order = [
    'Popularity', 'Standard MF', 'Causal MF',
    'Standard LightGCN', 'Causal LightGCN',
    'Causal Upper Bound'
]

colors = {
    'Popularity':         '#e74c3c',
    'Standard MF':        '#3498db',
    'Causal MF':          '#f39c12',
    'Standard LightGCN':  '#9b59b6',
    'Causal LightGCN':    '#2ecc71',
    'Causal Upper Bound': '#1abc9c'
}

# ─── FIGURE 1: MAIN RESULTS ───────────────
print("Creating Figure 1 — Main Results...")

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    'CausalRec-Bench — Experimental Results\n'
    'Category Precision@10 Across All Scenarios',
    fontsize=16, fontweight='bold', y=0.98
)

gs = gridspec.GridSpec(2, 3, figure=fig,
    hspace=0.45, wspace=0.35)

scenarios_to_plot = [
    ('Cold-Start', 'Cold-Start Users'),
    ('Level 3 - Hard', 'Level 3 Hard (All Confounders)'),
    ('Level 1 - Simple', 'Level 1 Simple (No Confounders)'),
    ('Winter Cold-Start', 'Winter Cold-Start'),
    ('E-commerce Domain', 'E-commerce Cold-Start'),
    ('Streaming Domain', 'Streaming Cold-Start'),
]

for i, (sc_key, sc_title) in enumerate(scenarios_to_plot):
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])

    sc_data = results[results['scenario']==sc_key]
    if len(sc_data) == 0:
        continue

    models = []
    values = []
    bar_colors = []

    for model in model_order:
        row_data = sc_data[sc_data['model']==model]
        if len(row_data) == 0:
            continue
        models.append(model.replace(' ', '\n'))
        values.append(row_data['category_p@10'].values[0])
        bar_colors.append(colors[model])

    bars = ax.bar(range(len(models)), values,
                  color=bar_colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    # Highlight best bar
    best_idx = np.argmax(values)
    bars[best_idx].set_edgecolor('#2c3e50')
    bars[best_idx].set_linewidth(2)

    # Add value labels
    for j, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    ax.set_title(sc_title, fontweight='bold',
                 fontsize=10, pad=8)
    ax.set_ylabel('Category Precision@10',
                  fontsize=8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=6.5)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.savefig('figures/main_results.png',
            dpi=150, bbox_inches='tight',
            facecolor='white')
print("  Saved figures/main_results.png")

# ─── FIGURE 2: KEY FINDINGS ───────────────
print("Creating Figure 2 — Key Findings...")

fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(
    'CausalRec-Bench — Key Findings',
    fontsize=15, fontweight='bold'
)

# Chart 1 — Finding 1: Causal MF improvement
ax1 = axes[0]
cold_data = results[results['scenario']=='Cold-Start']
mf_models = ['Popularity','Standard MF','Causal MF','Causal Upper Bound']
mf_vals = []
mf_cols = []
for m in mf_models:
    row = cold_data[cold_data['model']==m]
    if len(row) > 0:
        mf_vals.append(row['category_p@10'].values[0])
        mf_cols.append(colors[m])

bars1 = ax1.bar(
    [m.replace(' ','\n') for m in mf_models],
    mf_vals, color=mf_cols, alpha=0.85,
    edgecolor='white'
)
# Arrow showing improvement
std_val = cold_data[cold_data['model']=='Standard MF']['category_p@10'].values[0]
caus_val = cold_data[cold_data['model']=='Causal MF']['category_p@10'].values[0]
imp = (caus_val - std_val) / std_val * 100
ax1.annotate(
    f'+{imp:.1f}%',
    xy=(2, caus_val), xytext=(1.5, caus_val+0.05),
    fontsize=12, fontweight='bold', color='#2ecc71',
    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2)
)
ax1.set_title(
    f'Finding 1: Causal MF Cold-Start\n+{imp:.1f}% over Standard MF',
    fontweight='bold', fontsize=11
)
ax1.set_ylabel('Category Precision@10')
ax1.set_ylim(0, max(mf_vals)*1.3)
ax1.grid(True, alpha=0.3, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Chart 2 — Finding 2: LightGCN improvement
ax2 = axes[1]
l3_data = results[results['scenario']=='Level 3 - Hard']
lgcn_models = ['Standard LightGCN','Causal LightGCN','Causal Upper Bound']
lgcn_vals = []
lgcn_cols = []
for m in lgcn_models:
    row = l3_data[l3_data['model']==m]
    if len(row) > 0:
        lgcn_vals.append(row['category_p@10'].values[0])
        lgcn_cols.append(colors[m])

bars2 = ax2.bar(
    [m.replace(' ','\n') for m in lgcn_models],
    lgcn_vals, color=lgcn_cols, alpha=0.85,
    edgecolor='white'
)
std_l3 = l3_data[l3_data['model']=='Standard LightGCN']['category_p@10'].values[0]
caus_l3 = l3_data[l3_data['model']=='Causal LightGCN']['category_p@10'].values[0]
imp2 = (caus_l3 - std_l3) / std_l3 * 100
ax2.annotate(
    f'+{imp2:.1f}%',
    xy=(1, caus_l3), xytext=(0.5, caus_l3+0.05),
    fontsize=12, fontweight='bold', color='#2ecc71',
    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2)
)
ax2.set_title(
    f'Finding 2: Causal LightGCN Level 3\n+{imp2:.1f}% over Standard LightGCN',
    fontweight='bold', fontsize=11
)
ax2.set_ylabel('Category Precision@10')
ax2.set_ylim(0, max(lgcn_vals)*1.3)
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Chart 3 — Finding 3: Domain gap
ax3 = axes[2]
ecom_data = results[results['scenario']=='E-commerce Domain']
stream_data = results[results['scenario']=='Streaming Domain']

domain_models = ['Popularity','Standard MF','Causal MF','Standard LightGCN']
ecom_vals = []
stream_vals = []
for m in domain_models:
    e = ecom_data[ecom_data['model']==m]
    s = stream_data[stream_data['model']==m]
    ecom_vals.append(e['category_p@10'].values[0] if len(e)>0 else 0)
    stream_vals.append(s['category_p@10'].values[0] if len(s)>0 else 0)

x = np.arange(len(domain_models))
w = 0.35
ax3.bar(x-w/2, ecom_vals, w, label='E-commerce',
        color='#3498db', alpha=0.85)
ax3.bar(x+w/2, stream_vals, w, label='Streaming',
        color='#e67e22', alpha=0.85)

ax3.set_title(
    'Finding 3: Domain Gap\nE-commerce vs Streaming',
    fontweight='bold', fontsize=11
)
ax3.set_ylabel('Category Precision@10')
ax3.set_xticks(x)
ax3.set_xticklabels(
    [m.replace(' ','\n') for m in domain_models],
    fontsize=8
)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/key_findings.png',
            dpi=150, bbox_inches='tight',
            facecolor='white')
print("  Saved figures/key_findings.png")

# ─── FIGURE 3: VALIDATION ─────────────────
print("Creating Figure 3 — Benchmark Validation...")

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle(
    'CausalRec-Bench — Benchmark Validation',
    fontsize=14, fontweight='bold'
)

interactions = pd.read_csv('data/interactions.csv',
    usecols=['clicked','click_cause','avg_position',
             'promotion','popularity','domain'])

# Chart 1 — Click cause distribution
ax = axes3[0]
clicked = interactions[interactions['clicked']==True]
cause_counts = clicked['click_cause'].value_counts()
cause_labels = [c.replace('_',' ').title() for c in cause_counts.index]
cause_colors = ['#2ecc71','#e74c3c','#3498db','#f39c12','#9b59b6']
wedges, texts, autotexts = ax.pie(
    cause_counts.values,
    labels=cause_labels,
    colors=cause_colors[:len(cause_counts)],
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75
)
for text in autotexts:
    text.set_fontsize(8)
ax.set_title(
    'Click Cause Distribution\n(70% Non-Genuine)',
    fontweight='bold', fontsize=10
)

# Chart 2 — Position bias
ax2 = axes3[1]
pos_data = interactions.groupby('avg_position')['clicked'].mean()
ax2.bar(pos_data.index, pos_data.values,
        color='#3498db', alpha=0.8)
ax2.set_xlabel('Display Position')
ax2.set_ylabel('Click Rate')
ax2.set_title(
    f'Position Bias\n({pos_data.iloc[0]:.1%} vs {pos_data.iloc[-1]:.1%}, {pos_data.iloc[0]/pos_data.iloc[-1]:.2f}x)',
    fontweight='bold', fontsize=10
)
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Chart 3 — Promotion bias
ax3 = axes3[2]
promo_rate = interactions[interactions['promotion']=='promoted']['clicked'].mean()
nopromo_rate = interactions[interactions['promotion']=='not_promoted']['clicked'].mean()
ax3.bar(['Promoted','Not Promoted'],
        [promo_rate, nopromo_rate],
        color=['#e74c3c','#3498db'], alpha=0.85)
ax3.set_ylabel('Click Rate')
ax3.set_title(
    f'Promotion Bias\n({promo_rate:.1%} vs {nopromo_rate:.1%}, {promo_rate/nopromo_rate:.2f}x)',
    fontweight='bold', fontsize=10
)
for i, v in enumerate([promo_rate, nopromo_rate]):
    ax3.text(i, v+0.003, f'{v:.1%}',
             ha='center', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/validation.png',
            dpi=150, bbox_inches='tight',
            facecolor='white')
print("  Saved figures/validation.png")

print()
print("=" * 55)
print("ALL CHARTS GENERATED")
print("=" * 55)
print()
print("Figures saved:")
print("  figures/main_results.png")
print("  figures/key_findings.png")
print("  figures/validation.png")
