"""Figure H: label-space mismatch dominates the cross-cohort transfer gap."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.family':'serif','font.size':8,'savefig.dpi':300,
                     'savefig.bbox':'tight','axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':0.7})

R = Path('/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer')
dec = json.loads((R/'results/route_a/shift_decomposition.json').read_text())
simj = json.loads((R/'results/route_a/sim_mismatch.json').read_text())
sim, n_seeds = simj['rows'], simj['design']['n_seeds']
classes = dec['class_order']

C_FOCAL, C_BASE, C_NOMASK = '#1f5fa8', '#9a9a9a', '#e08a1e'
C_ALARM, C_LOW = '#c0392b', '#c9dcf0'

fig = plt.figure(figsize=(7.3, 2.75))
gs = fig.add_gridspec(1, 3, wspace=0.46)

# ---------------- Panel A ----------------
axA = fig.add_subplot(gs[0, 0])
cohs = ['TCGA', 'GSE21653']
labels = ['TCGA-BRCA\n(n=987)', 'GSE21653\n(n=266)']
stages = [('base', 'uncorrected', C_BASE),
          ('mask_only', 'label space\nreconciled', C_LOW),
          ('em_pam50', 'reconciled\n+ prior adjust', C_FOCAL)]
x = np.arange(len(cohs)); w = 0.26
for i, (key, lab, col) in enumerate(stages):
    vals = [dec['cohorts'][c][key]['acc'] * 100 for c in cohs]
    b = axA.bar(x + (i - 1) * w, vals, w, color=col, edgecolor='white',
                linewidth=0.5, label=lab, zorder=2)
    if i != 1:
        for r, v in zip(b, vals):
            axA.text(r.get_x() + r.get_width()/2, v + 0.8, f'{v:.1f}',
                     ha='center', va='bottom', fontsize=6.2, zorder=4,
                     color=C_FOCAL if i == 2 else '#444444',
                     fontweight='bold' if i == 2 else 'normal')
# gain annotation above the corrected bar
for j, c in enumerate(cohs):
    g = (dec['cohorts'][c]['em_pam50']['acc'] - dec['cohorts'][c]['base']['acc']) * 100
    axA.text(x[j] + w, 85.4, f'$\\Delta$ +{g:.1f} pts', ha='center', va='bottom',
             fontsize=6.4, color=C_FOCAL, fontweight='bold', zorder=4)
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=6.8)
axA.set_ylabel('Accuracy (%)')
axA.set_ylim(55, 90.5); axA.set_xlim(-0.52, 1.52)
axA.set_title('Reconciling the label space\nrecovers most of the gap',
              fontsize=7.6, loc='left', pad=6)
axA.legend(fontsize=5.9, frameon=False, loc='upper center',
           bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength=0.9,
           columnspacing=0.9, handletextpad=0.45, borderpad=0.0)

# ---------------- Panel B ----------------
axB = fig.add_subplot(gs[0, 1])
est = np.array(dec['cohorts']['TCGA']['em_full_prior']) * 100
tru = np.array(dec['cohorts']['TCGA']['true_prior']) * 100
xm = np.arange(len(classes)); w2 = 0.38
cols = [C_ALARM if c == 'claudin-low' else '#6b9bd1' for c in classes]
axB.bar(xm - w2/2, est, w2, color=cols, edgecolor='white', linewidth=0.5,
        label='EM estimate (no mask)', zorder=2)
axB.bar(xm + w2/2, tru, w2, color='#d9d9d9', edgecolor='white', linewidth=0.5,
        label='true prevalence', zorder=2)
axB.axhline(0, color='#333333', lw=0.6, zorder=3)
i_cl = classes.index('claudin-low')
axB.annotate(f'{est[i_cl]:.1f}% on a class\nthat cannot occur',
             xy=(i_cl - w2/2, est[i_cl] + 1.0), xytext=(2.62, 36),
             fontsize=6.2, color=C_ALARM, ha='left', va='center', zorder=5,
             arrowprops=dict(arrowstyle='->', color=C_ALARM, lw=0.75,
                             shrinkA=2, shrinkB=2,
                             connectionstyle='arc3,rad=-0.12'))
axB.text(0.03, 0.99, 'unmasked EM recovers\nonly 2.2 of 6.9 points',
         transform=axB.transAxes, fontsize=6.2, va='top', ha='left',
         color='#444444', zorder=5)
axB.set_xticks(xm)
axB.set_xticklabels([c.replace('claudin-low', 'clau.').replace('Normal', 'Norm.') for c in classes],
                    fontsize=5.9, rotation=48, ha='right')
axB.set_ylabel('Prior mass (%)')
axB.set_ylim(0, 66); axB.set_xlim(-0.55, 5.55)
axB.set_title('Unmasked EM keeps mass\non an impossible class',
              fontsize=7.6, loc='left', pad=6)
axB.legend(fontsize=5.9, frameon=False, loc='upper center',
           bbox_to_anchor=(0.5, -0.42), ncol=1, handlelength=0.9,
           handletextpad=0.45, borderpad=0.0, labelspacing=0.25)

# ---------------- Panel C ----------------
axC = fig.add_subplot(gs[0, 2])
for ps, col, lab, mk in [(True, C_FOCAL, 'class priors also shift', 'o'),
                         (False, C_NOMASK, 'label space differs only', 's')]:
    xs, ys = [], []
    for r in sim:
        if r['prior_shift'] == ps:
            xs.append(r['n_absent'] + (r['cov_shift'] - 0.4) * 0.55)
            ys.append(r['gain_base_to_detected'] * 100)
    axC.scatter(xs, ys, s=17, color=col, marker=mk, alpha=0.9, zorder=3,
                edgecolor='white', linewidth=0.4, label=lab)
axC.axhline(0, color='#333333', lw=0.7, zorder=1)
axC.set_xticks([0, 1, 2]); axC.set_xticklabels(['0', '1', '2'], fontsize=6.8)
axC.set_xlabel('absent classes in target', labelpad=2)
axC.set_ylabel('Accuracy gain (points)')
axC.set_ylim(-4.2, 10.2); axC.set_xlim(-0.45, 2.5)
axC.set_title('The gain needs both mismatches,\nnot a label-space mismatch alone',
              fontsize=7.6, loc='left', pad=6)
axC.legend(fontsize=5.9, frameon=False, loc='lower center', ncol=1,
           handlelength=0.9, handletextpad=0.45, borderpad=0.0,
           labelspacing=0.25, markerscale=1.1)

for ax, L in zip([axA, axB, axC], 'ABC'):
    ax.text(-0.32, 1.20, L, transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', va='top', ha='left')

fig.savefig(str(R/'paper/figures/figH_label_shift.pdf'), bbox_inches='tight')
fig.savefig('/tmp/figH_label_shift.png', dpi=300, bbox_inches='tight')
print(f'saved figH  (n_seeds={n_seeds})')
