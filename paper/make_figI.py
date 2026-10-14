"""Figure I: treatment-response (pCR) discrimination in GSE20194."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.family':'serif','font.size':8,'savefig.dpi':300,
                     'savefig.bbox':'tight','axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':0.7})

R = Path('/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer')
d = json.loads((R/'results/route_a/pcr_gse20194.json').read_text())
C_CLIN, C_GEN, C_COMB = '#7f7f7f', '#e08a1e', '#1f5fa8'
C_HIGH, C_MID, C_LOW = '#c0392b', '#e08a1e', '#8fa8c8'

fig = plt.figure(figsize=(7.0, 2.75))
gs = fig.add_gridspec(1, 2, wspace=0.34, width_ratios=[1.0, 1.18])

# ---------------- Panel A: AUCs, no increment ----------------
axA = fig.add_subplot(gs[0, 0])
mk = ['Clinical\nonly', 'Genomic\nonly', 'Clinical +\ngenomic']
vals = [d['clinical_only']['auc'], d['genomic_only']['auc'],
        d['clinical_plus_genomic']['auc']]
sds  = [d['clinical_only']['auc_std'], d['genomic_only']['auc_std'],
        d['clinical_plus_genomic']['auc_std']]
cols = [C_CLIN, C_GEN, C_COMB]
b = axA.bar(range(3), vals, 0.6, color=cols, edgecolor='white', linewidth=0.6,
            yerr=sds, capsize=2.5, error_kw=dict(lw=0.7, ecolor='#555555'), zorder=2)
for r, v in zip(b, vals):
    axA.text(r.get_x()+r.get_width()/2, v+0.012, f'{v:.3f}', ha='center',
             va='bottom', fontsize=6.6, fontweight='bold', color='#333333')
axA.axhline(0.5, color='#333333', lw=0.7, ls=(0,(4,3)), zorder=1)
axA.text(-0.45, 0.512, 'chance', fontsize=5.8, color='#777777', ha='left')
# the non-significant delta, placed inside the axes
dl = d['delta_genomic_over_clinical']
lo, hi = dl['ci95']
axA.text(0.97, 0.96,
         f'genomic over clinical:\n$\\Delta$={dl["mean"]:+.4f}\n95% CI [{lo:+.3f}, {hi:+.3f}]',
         transform=axA.transAxes, fontsize=5.9, color='#c0392b',
         va='top', ha='right', linespacing=1.35, zorder=5)
axA.set_xticks(range(3)); axA.set_xticklabels(mk, fontsize=6.6)
axA.set_ylabel('AUC (pCR vs residual disease)')
axA.set_ylim(0.45, 1.06); axA.set_xlim(-0.55, 2.55)
axA.set_title('Adding genomics to clinical\nvariables does not help', fontsize=7.6, loc='left', pad=6)

# ---------------- Panel B: pCR rate by predicted subtype ----------------
axB = fig.add_subplot(gs[0, 1])
order = ['LumA', 'LumB', 'Normal', 'claudin-low', 'Basal', 'Her2']
nice  = {'LumA':'Luminal A', 'LumB':'Luminal B', 'Normal':'Normal-like',
         'claudin-low':'Claudin-low', 'Basal':'Basal-like', 'Her2':'HER2-enr.'}
sp = d['predicted_subtype_pcr']
rates = [sp[k]['rate']*100 for k in order]
ns    = [sp[k]['n'] for k in order]
ks    = [sp[k]['pcr'] for k in order]
colB  = [C_LOW if k in ('LumA','LumB','Normal') else (C_MID if k=='claudin-low' else C_HIGH)
         for k in order]
y = np.arange(len(order))[::-1]
axB.barh(y, rates, 0.62, color=colB, edgecolor='white', linewidth=0.6, zorder=2)
for yy, r, n, k in zip(y, rates, ns, ks):
    axB.text(r+1.1, yy, f'{r:.1f}%  ({k}/{n})', va='center', ha='left',
             fontsize=6.3, color='#333333')
overall = d['n_pcr'] / d['n'] * 100
axB.axvline(overall, color='#333333', lw=0.8, ls=(0,(4,3)), zorder=3)
axB.text(overall+0.9, y[0]+0.42, f'cohort mean {overall:.1f}%', fontsize=6.0,
         color='#333333', va='center', ha='left')
axB.set_yticks(y); axB.set_yticklabels([nice[k] for k in order], fontsize=6.8)
axB.set_xlabel('pCR rate (%)')
axB.set_xlim(0, 78); axB.set_ylim(-0.6, len(order)-0.15)
axB.set_title('Predicted subtype still recovers\nthe expected response biology',
              fontsize=7.6, loc='left', pad=6)

for ax, L in zip([axA, axB], 'AB'):
    ax.text(-0.30, 1.20, L, transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', va='top', ha='left')

fig.savefig(str(R/'paper/figures/figI_pcr.pdf'), bbox_inches='tight')
fig.savefig('/tmp/figI_pcr.png', dpi=300, bbox_inches='tight')
print(f'saved figI_pcr.pdf  (n={d["n"]}, pCR={d["n_pcr"]}, RD={d["n_rd"]})')
