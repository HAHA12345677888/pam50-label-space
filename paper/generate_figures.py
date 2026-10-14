"""
Generate publication-quality figures for the multimodal breast cancer subtyping paper.
Run: python paper/generate_figures.py
Output: paper/figures/*.pdf
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# --- Style config ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SUBTYPES = ['LumA', 'LumB', 'HER2', 'CL', 'Basal', 'Normal']
COLORS = {
    'LumA': '#1f77b4', 'LumB': '#ff7f0e', 'Her2': '#2ca02c',
    'Claudin-low': '#d62728', 'Basal': '#9467bd', 'Normal': '#8c564b',
}
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def make_figures_dir():
    os.makedirs('paper/figures', exist_ok=True)


FIG_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'results', 'route_a', 'figure_data.json')


def load_figure_data():
    """Measured values from scripts/route_a_fix_figures.py.

    Every number plotted in Figures 2 and 3 comes from this file. Nothing in
    those figures is drawn from a hand-built or schematic matrix.
    """
    with open(FIG_DATA) as fh:
        return json.load(fh)


def annotate_cm(ax, cm, thresh):
    """Write count and row-percentage into each cell."""
    for i in range(cm.shape[0]):
        rtot = cm[i].sum()
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = 'white' if v > thresh else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i - 0.13, str(v), ha='center', va='center',
                    fontsize=9, color=color, fontweight=weight)
            ax.text(j, i + 0.22, f'{100 * v / rtot:.0f}%', ha='center', va='center',
                    fontsize=7, color=color)


# ============================================================
# Figure 1: Modality Ablation Bar Chart
# ============================================================
def fig_modality_ablation():
    combos = [
        'Clinical\nonly', 'mRNA\nonly', 'Clin+mRNA', 'Clin+mRNA\n+Methyl',
        '+CNA\n(4-mod)', '+Mut\n(4-mod)', 'All 5\nmodalities',
    ]
    cv_vals = [64.1, 73.8, 74.5, 75.3, 73.0, 73.3, 72.4]
    cv_std  = [2.7,  1.6,  1.9,  2.6,  1.8,  2.0,  2.0]
    dims    = [69, 412, 481, 713, 969, 886, 1142]
    colors  = ['#aec7e8', '#98df8a', '#c5b0d5', '#1f77b4', '#c49c94', '#f7b6d2', '#c7c7c7']

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(range(len(combos)), cv_vals, yerr=cv_std, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.5, width=0.7)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, fontsize=8)
    ax.set_ylabel('5-fold CV Accuracy (%)')
    ax.set_ylim(55, 82)
    ax.axhline(y=75.3, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Best (75.3%)')
    for bar, dim in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3.0,
                f'{dim}d', ha='center', fontsize=7, color='#555555')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig('paper/figures/fig1_modality_ablation.pdf')
    plt.close(fig)
    print('Saved fig1_modality_ablation.pdf')


# ============================================================
# Figure 2: METABRIC Test Confusion Matrix
# ============================================================
def fig_confusion_metabric():
    """Figure 2: confusion matrix of the 3-modality model on the METABRIC test set.

    Values are the measured predictions of LogisticRegression(C=0.01) trained on
    Clinical+mRNA+Methylation (713 dimensions) over data/train.npz and evaluated
    on data/test.npz -- see scripts/route_a_fix_figures.py.
    """
    md = load_figure_data()['metabric_test']
    cm = np.array(md['confusion'], dtype=int)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(SUBTYPES, fontsize=8, rotation=30, ha='right')
    ax.set_yticklabels(SUBTYPES, fontsize=8)
    ax.set_xlabel('Predicted Subtype')
    ax.set_ylabel('True Subtype')
    annotate_cm(ax, cm, thresh=cm.max() * 0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Samples')
    ax.set_title(f'METABRIC test set (n={md["n"]}, '
                 f'accuracy={md["accuracy"] * 100:.1f}%)', fontsize=10)
    fig.tight_layout()
    fig.savefig('paper/figures/fig2_confusion_metabric.pdf')
    plt.close(fig)
    print(f'Saved fig2_confusion_metabric.pdf  (n={md["n"]}, '
          f'acc={md["accuracy"] * 100:.2f}%)')


# ============================================================
# Figure 3: TCGA External Validation
# ============================================================
def fig_tcga_results():
    """Figure 3: TCGA-BRCA external validation.

    Panel A is the measured confusion matrix of the mRNA-only linear reference
    on TCGA (5 true classes present, 6 possible predictions). Panel B is the
    Kaplan-Meier estimate of the two median-split risk groups, fitted on the
    actual TCGA overall-survival outcomes. Both come from
    scripts/route_a_fix_figures.py.
    """
    fd = load_figure_data()
    td = fd['tcga_logreg']
    km = fd['tcga_km']

    cm = np.array(td['confusion'], dtype=int)          # 5 true rows x 6 predictions
    col_lab = SUBTYPES                                  # all six predictions possible
    row_lab = ['LumA', 'LumB', 'HER2', 'Basal', 'Normal']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: measured confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap='Oranges', interpolation='nearest', aspect='auto')
    ax.set_xticks(range(len(col_lab)))
    ax.set_yticks(range(len(row_lab)))
    ax.set_xticklabels(col_lab, fontsize=7.5, rotation=35, ha='right')
    ax.set_yticklabels(row_lab, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    annotate_cm(ax, cm, thresh=cm.max() * 0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Samples')
    ax.set_title(f'A. TCGA-BRCA confusion (n={td["n"]}, '
                 f'accuracy={td["accuracy"] * 100:.1f}%)',
                 fontsize=10, loc='left', fontweight='bold')

    # Panel B: real Kaplan-Meier curves
    ax2 = axes[1]
    style = {'low': ('#1f77b4', 'Low risk'), 'high': ('#d62728', 'High risk')}
    for grp in ('low', 'high'):
        c, label = style[grp]
        cur = km['curves'][grp]
        ax2.step(cur['timeline'], cur['survival'], where='post',
                 color=c, linewidth=1.6,
                 label=f'{label} (n={cur["n"]}, {cur["events"]} events)')
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Overall survival probability')
    ax2.set_title(f'B. TCGA-BRCA Kaplan-Meier '
                  f'(C-index={km["c_index"]:.3f}, log-rank $p$={km["logrank_p"]:.3f})',
                  fontsize=10, loc='left', fontweight='bold')
    ax2.legend(loc='lower left', frameon=False, fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig('paper/figures/fig3_tcga_results.pdf')
    plt.close(fig)
    n_cl = int(cm[:, 3].sum())
    print(f'Saved fig3_tcga_results.pdf  (acc={td["accuracy"] * 100:.2f}%, '
          f'C-index={km["c_index"]:.4f}, {n_cl} samples predicted claudin-low)')


# ============================================================
# Figure 4: SHAP Analysis
# ============================================================
def fig_shap_analysis():
    """Two-panel: modality pie + top features bar."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Modality pie
    ax = axes[0]
    mods = ['mRNA\n(63.3%)', 'Methylation\n(26.7%)', 'Clinical\n(10.0%)']
    sizes = [63.3, 26.7, 10.0]
    cols = ['#1f77b4', '#2ca02c', '#ff7f0e']
    wedges, texts = ax.pie(sizes, labels=mods, colors=cols, startangle=90,
                           textprops={'fontsize': 9})
    ax.set_title('A. Modality Contribution', fontsize=10, loc='left', fontweight='bold')

    # Panel B: Top-10 features bar
    ax2 = axes[1]
    features = [
        'THREEGENE_ER+/HER2-\nHigh Prolif', 'THREEGENE_ER+/HER2-\nLow Prolif',
        'ER_STATUS\nPositive', 'ER_STATUS\nNegative', 'ACP7',
        'THREEGENE\nER-/HER2-', 'ACTG2', 'ACOT4', 'ADM', 'ADCY3',
    ]
    shap_vals = [0.0614, 0.0579, 0.0511, 0.0511, 0.0505,
                 0.0473, 0.0467, 0.0436, 0.0425, 0.0416]
    modality_colors = {
        'Clinical': '#ff7f0e', 'mRNA': '#1f77b4', 'Methylation': '#2ca02c',
    }
    feat_mods = ['Clinical'] * 4 + ['mRNA'] * 6
    colors_bar = [modality_colors[m] for m in feat_mods]
    y_pos = range(len(features))
    ax2.barh(y_pos, shap_vals, color=colors_bar, edgecolor='black', linewidth=0.3, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features, fontsize=7.5)
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.set_title('B. Top 10 Features by SHAP', fontsize=10, loc='left', fontweight='bold')
    ax2.invert_yaxis()
    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in modality_colors.items()]
    ax2.legend(handles=patches, loc='lower right', frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig('paper/figures/fig4_shap_analysis.pdf')
    plt.close(fig)
    print('Saved fig4_shap_analysis.pdf')


# ============================================================
# Figure 5: Model Comparison
# ============================================================
def fig_model_comparison():
    models = [
        'Majority', 'LR (Clin)', 'LR (mRNA)', 'RF', 'XGBoost',
        'LightGBM', 'CatBoost', 'ET', 'Soft-vote', 'Transformer',
        'Guided TF', 'Cascade',
    ]
    cv = [35.6, 64.1, 73.8, 72.4, 72.0, 71.0, 73.9, 71.3, 72.6, 68.5, 74.0, 76.2]
    std = [0, 2.7, 1.6, 2.4, 3.3, 3.1, 2.7, 1.7, 2.8, 1.8, 1.0, 2.0]
    colors = ['#c7c7c7'] + ['#aec7e8'] * 2 + ['#98df8a'] * 6 + ['#c5b0d5'] * 3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(models)), cv, xerr=std, color=colors,
                   edgecolor='black', linewidth=0.4, height=0.7, capsize=3)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8.5)
    ax.set_xlabel('5-fold CV Accuracy (%)')
    ax.set_xlim(30, 82)
    ax.axvline(x=75.3, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='LR best (75.3%)')
    ax.invert_yaxis()
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig('paper/figures/fig5_model_comparison.pdf')
    plt.close(fig)
    print('Saved fig5_model_comparison.pdf')


# ============================================================
# Figure 6: Cross-Platform Validation Summary
# ============================================================
def fig_cross_platform():
    """All three cohorts evaluated with the SAME estimator: the mRNA-only model
    (412 genes, LogReg C=0.01, within-platform rank normalization). Values are read
    from the measured run rather than hardcoded, so the figure cannot drift from the
    manuscript. TCGA is mRNA-only by necessity (no METABRIC-compatible clinical or
    methylation blocks), so the multi-modality METABRIC numbers are NOT plotted here."""
    import json, os
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                      'results', 'route_a', 'mrna_only_crossplatform.json')
    _d = json.load(open(_p))
    fig, ax = plt.subplots(figsize=(6, 4))
    datasets = ['METABRIC\nValidation', 'METABRIC\nTest', 'TCGA-BRCA\n(External)']
    acc = [_d['val']['acc'] * 100, _d['test']['acc'] * 100, _d['tcga']['acc'] * 100]
    f1  = [_d['val']['f1'], _d['test']['f1'], _d['tcga']['f1']]
    x = np.arange(len(datasets))
    width = 0.32
    bars1 = ax.bar(x - width/2, acc, width, label='Accuracy (%)', color='#1f77b4', edgecolor='black', linewidth=0.4)
    bars2 = ax.bar(x + width/2, [f*100 for f in f1], width, label='Macro F1 (%)', color='#ff7f0e', edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(40, 90)
    ax.legend(frameon=False)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%',
                ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%',
                ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig('paper/figures/fig6_cross_platform.pdf')
    plt.close(fig)
    print('Saved fig6_cross_platform.pdf')


# ============================================================
if __name__ == '__main__':
    make_figures_dir()
    fig_modality_ablation()
    fig_confusion_metabric()
    fig_tcga_results()
    fig_shap_analysis()
    fig_model_comparison()
    fig_cross_platform()
    print('\nAll 6 figures saved to paper/figures/')
