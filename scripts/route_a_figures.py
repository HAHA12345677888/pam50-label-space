"""Figures B-E for the Route A section (A and F are produced by their own scripts)."""
import json, os, sys
sys.path.insert(0, "/tmp"); exec(open("/tmp/fig_style.py").read())
import matplotlib.pyplot as plt, numpy as np

R = "results/route_a"; F = "figures"
os.makedirs(F, exist_ok=True)
REF = 0.7526  # LogReg reference CV

# ---------------- figB: capacity sweep ----------------
d = json.load(open(f"{R}/diagnosis.json"))
cap = d["capacity_sweep"]
keys = sorted(cap, key=lambda k: ([int(x) for x in k.replace("d", "").replace("l", "").split("_")
                                   if x.isdigit()] if False else 0))
def parse(k):
    p = k.split("_"); return (int(p[0][1:]), int(p[1][1:]))
keys = sorted(cap, key=parse)
vals = [cap[k]["cv"] * 100 for k in keys]; errs = [cap[k]["std"] * 100 for k in keys]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
x = np.arange(len(keys))
ax.errorbar(x, vals, yerr=errs, fmt="o", ms=4, capsize=3, color="#1f4e79", lw=1)
ax.axhline(REF * 100, ls="--", lw=1.2, color="#c00000")
ax.text(0.02, REF * 100 + 0.25, f"LogReg reference ({REF*100:.1f}%)", color="#c00000",
        fontsize=7, va="bottom")
ax.set_xticks(x); ax.set_xticklabels([f"d{a}\nL{b}" for a, b in map(parse, keys)], fontsize=6)
ax.set_ylabel("5-fold CV accuracy (%)"); ax.set_ylim(62, 78)
ax.set_title("Capacity sweep: 12 configurations", loc="left", fontweight="bold", fontsize=9)
plt.tight_layout(); fig.savefig(f"{F}/figB_capacity_sweep.pdf"); fig.savefig(f"{F}/figB_capacity_sweep.png")
print("saved figB")

# ---------------- figC: regularization ----------------
d = json.load(open(f"{R}/regularized_cv.json"))
sa = d["single_axis"]
order = ["baseline", "mod_drop_0.1", "mod_drop_0.2", "mod_drop_0.3", "mod_drop_0.5",
         "wd_0.05", "wd_0.1", "label_smooth_0.1", "dropout_0.5", "d32_l1", "d32_l2_moddrop"]
order = [k for k in order if k in sa]
labs = [k.replace("mod_drop", "MD").replace("label_smooth", "LS").replace("_", " ") for k in order]
vals = [sa[k]["cv"] * 100 for k in order]; errs = [sa[k]["std"] * 100 for k in order]
fig, ax = plt.subplots(figsize=(5.4, 2.9))
cols = ["#7f7f7f"] + ["#1f4e79"] * (len(order) - 1)
ax.bar(range(len(order)), vals, yerr=errs, capsize=2.5, color=cols)
ax.axhline(REF * 100, ls="--", lw=1.2, color="#c00000")
ax.text(0.02, REF * 100 + 0.3, f"LogReg reference ({REF*100:.1f}%)", color="#c00000", fontsize=7)
ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=6)
ax.set_ylabel("5-fold CV accuracy (%)"); ax.set_ylim(62, 78)
ax.set_title("Regularization axes (single-axis, d64 L2)", loc="left", fontweight="bold", fontsize=9)
plt.tight_layout(); fig.savefig(f"{F}/figC_regularization.pdf"); fig.savefig(f"{F}/figC_regularization.png")
print("saved figC")

# ---------------- figD: pretraining ----------------
d = json.load(open(f"{R}/pretrain_cv.json"))
labels = ["Scratch", "Masked-feature\npretrained", "LogReg\n(reference)"]
vals = [d["scratch"]["cv"] * 100, d["pretrained"]["cv"] * 100, REF * 100]
errs = [d["scratch"]["std"] * 100, d["pretrained"]["std"] * 100, 0]
fig, ax = plt.subplots(figsize=(3.9, 2.9))
ax.bar(range(3), vals, yerr=errs, capsize=3, color=["#a6a6a6", "#1f4e79", "#c00000"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("5-fold CV accuracy (%)"); ax.set_ylim(60, 79)
ax.set_title("Self-supervised pretraining", loc="left", fontweight="bold", fontsize=9)
plt.tight_layout(); fig.savefig(f"{F}/figD_pretraining.pdf"); fig.savefig(f"{F}/figD_pretraining.png")
print("saved figD")

# ---------------- figE: TCGA validation, CV vs external ----------------
tb = json.load(open(f"{R}/tcga_best.json"))
sn = json.load(open(f"{R}/tcga_survival.json"))
names = ["LogReg", "Transformer", "TF + pretraining"]
cv = [REF * 100, d["scratch"]["cv"] * 100, d["pretrained"]["cv"] * 100]
tc = [tb["logreg"]["acc"] * 100, tb["tf"]["acc"] * 100, tb["tf_pre"]["acc"] * 100]
x = np.arange(3); w = 0.36
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
a = axes[0]
a.bar(x - w/2, cv, w, color="#95b3d7", label="METABRIC CV")
a.bar(x + w/2, tc, w, color="#1f4e79", label="TCGA external")
for i in range(3):
    a.annotate("", xy=(i + w/2, tc[i]), xytext=(i - w/2, cv[i]),
               arrowprops=dict(arrowstyle="->", lw=1,
                               color=("#c00000" if tc[i] < cv[i] else "#2e7d32")))
a.set_xticks(x); a.set_xticklabels(names, fontsize=7)
a.set_ylabel("Accuracy (%)"); a.set_ylim(55, 82); a.legend(fontsize=7, loc="lower left")
a.set_title("Internal vs external accuracy", loc="left", fontweight="bold", fontsize=9)
b = axes[1]
ci = [sn["logreg"]["c_index"], sn["tf"]["c_index"], sn["tf_pre"]["c_index"]]
b.bar(x, ci, 0.5, color=["#c00000", "#1f4e79", "#1f4e79"])
b.axhline(0.5, ls=":", lw=1, color="k")
b.axhline(sn["published_c_index"], ls="--", lw=1, color="#7f7f7f")
b.text(2.4, sn["published_c_index"] + 0.004, "published\n0.586", fontsize=6,
       color="#7f7f7f", ha="right")
for i, v in enumerate(ci):
    b.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
b.set_xticks(x); b.set_xticklabels(names, fontsize=7)
b.set_ylabel("TCGA C-index"); b.set_ylim(0.45, 0.63)
b.set_title("External survival discrimination", loc="left", fontweight="bold", fontsize=9)
plt.tight_layout(); fig.savefig(f"{F}/figE_tcga_validation.pdf"); fig.savefig(f"{F}/figE_tcga_validation.png")
print("saved figE")
print("\nCV -> TCGA deltas:")
for n_, c_, t_ in zip(names, cv, tc):
    print(f"  {n_:18s} CV={c_:.2f}  TCGA={t_:.2f}  delta={t_-c_:+.2f}")
