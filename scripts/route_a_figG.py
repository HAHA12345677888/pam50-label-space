"""figG: second external cohort + the multi-cohort training test."""
import json, sys
sys.path.insert(0, "/tmp"); exec(open("/tmp/fig_style.py").read())
import matplotlib.pyplot as plt, numpy as np

R="results/route_a"
g=json.load(open(f"{R}/gse21653_validation.json"))
mc=json.load(open(f"{R}/multicohort_cv.json"))
tb=json.load(open(f"{R}/tcga_best.json"))

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

# (a) per-cohort external accuracy + survival C-index
a = axes[0]
coh = ["METABRIC\n(CV)", "TCGA-BRCA\n(n=987)", "GSE21653\n(n=266)"]
acc = [0.7526*100, tb["logreg"]["acc"]*100, g["accuracy"]*100]
a.bar(range(3), acc, 0.55, color=["#95b3d7", "#1f4e79", "#2e75b6"])
for i, v in enumerate(acc):
    a.text(i, v+0.8, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
a.set_xticks(range(3)); a.set_xticklabels(coh, fontsize=7)
a.set_ylabel("Accuracy (%)"); a.set_ylim(0, 88)
a.set_title("(a) Linear reference across cohorts", loc="left", fontweight="bold", fontsize=8.5)

# (b) survival discrimination in the two external cohorts
b = axes[1]
sn = json.load(open(f"{R}/tcga_survival.json"))
ci = [sn["logreg"]["c_index"], g["survival"]["c_index"]]
b.bar([0, 1], ci, 0.5, color=["#1f4e79", "#2e75b6"])
b.axhline(0.5, ls=":", lw=1, color="k")
for i, v in enumerate(ci):
    b.text(i, v+0.006, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
b.set_xticks([0, 1]); b.set_xticklabels(["TCGA-BRCA\n(n=966)", "GSE21653\n(n=252)"], fontsize=7)
b.set_ylabel("External C-index"); b.set_ylim(0.45, 0.65)
b.set_title("(b) Survival discrimination", loc="left", fontweight="bold", fontsize=8.5)

# (c) multi-cohort training test vs prediction
c = axes[2]
mo, mg = mc["results"]["metabric_only"], mc["results"]["metabric+gse21653"]
names = ["LogReg", "Transformer", "TF+pretrain"]
d = [100*(mg[m]["acc"]-mo[m]["acc"]) for m in ["logreg","tf","tf_pre"]]
c.bar(range(3), d, 0.5, color=["#a6a6a6", "#1f4e79", "#1f4e79"])
c.axhline(0, lw=1, color="k")
c.axhline(1.00, ls="--", lw=1.2, color="#c00000")
c.text(2.45, 1.08, "learning-curve\nprediction (+1.0)", fontsize=6, color="#c00000", ha="right")
for i, v in enumerate(d):
    c.text(i, v + (0.06 if v > 0 else -0.14), f"{v:+.2f}", ha="center", fontsize=8, fontweight="bold")
c.set_xticks(range(3)); c.set_xticklabels(names, fontsize=7)
c.set_ylabel("Δ TCGA accuracy (points)"); c.set_ylim(-1.5, 1.6)
c.set_title("(c) Effect of adding GSE21653 to training", loc="left", fontweight="bold", fontsize=8.5)

plt.tight_layout()
fig.savefig("figures/figG_second_cohort.pdf"); fig.savefig("figures/figG_second_cohort.png")
print("saved figG")
