"""
Step 7b: does adding a second cohort as TRAINING data narrow the Transformer gap?

Step 2 concluded the gap is a bias limitation that would need roughly 6x more data to
close. This tests the prediction directly with real additional samples: METABRIC train
(n=958) + GSE21653 (n=266) = 1224 training samples, evaluated on TCGA (n=987) so the
external endpoint stays fixed and comparable.

Restricted to the 369 panel genes mapped on both Affymetrix (GPL570) and the METABRIC
panel, trained mRNA-only, since GSE21653 has no clinical or methylation blocks.
Each platform is rank-normalized independently before pooling.
"""
import json, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_tf import LightTransformer, train_one, predict
from route_a_step4_pretrain import pretrain
from route_a_step5_tcga import rank_norm
from route_a_step7_gse21653 import load_gse, map_probes
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ---- assemble the three platforms on a common gene set ----
mods, ytr, _, _, _ = pool_splits(["train"])
genes = None
import glob
for c in [os.path.join(ROOT, "feature_names.json")] + glob.glob(os.path.join(ROOT, "data/*.json")):
    if os.path.exists(c):
        j = json.load(open(c))
        if isinstance(j, dict) and "mrna_genes" in j:
            genes = [g.strip() for g in j["mrna_genes"]]; break
probes, Xg, y_g, dur_g, ev_g, _ = load_gse()
pmap = map_probes(probes, Xg)
found = [g for g in genes if g in pmap]
cols = [genes.index(g) for g in found]
print(f"common gene set: {len(found)} genes")

M_tr = np.nan_to_num(mods[1][:, cols], nan=0.0).astype(np.float64)
G = np.nan_to_num(Xg[[pmap[g] for g in found]].T, nan=0.0).astype(np.float64)
d_tc = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
# TCGA uses the full 412-gene panel; align it to the common set
genes_full = genes
tc_idx = [genes_full.index(g) for g in found]
T = d_tc["mrna"][:, tc_idx].astype(np.float64)
y_tc = d_tc["labels"]

M_tr, G, T = rank_norm(M_tr), rank_norm(G), rank_norm(T)

results = {}
for name, Xtr, ytr_ in [
    ("metabric_only", M_tr, ytr),
    ("metabric+gse21653", np.vstack([M_tr, G]), np.concatenate([ytr, y_g])),
]:
    print(f"\n--- {name}: n_train={len(ytr_)} ---")
    res = {"n_train": int(len(ytr_))}
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr_)
    p = lr.predict(sc.transform(T))
    res["logreg"] = {"acc": float(accuracy_score(y_tc, p)),
                     "macro_f1": float(f1_score(y_tc, p, average="macro"))}
    print(f"  logreg : {res['logreg']['acc']*100:.2f}%")
    for mname, pre in [("tf", False), ("tf_pre", True)]:
        Xtr32, T32 = Xtr.astype(np.float32), T.astype(np.float32)
        m = LightTransformer([Xtr.shape[1]], d_model=64, n_layers=2, nhead=4, dim_ff=128, dropout=0.3)
        if pre:
            m = pretrain([torch.tensor(Xtr32)], [Xtr.shape[1]]).transfer_to(m)
        train_one(m, [Xtr32], ytr_, None, epochs=80, seed=SEED)
        pp = predict(m, [T32]).argmax(1)
        res[mname] = {"acc": float(accuracy_score(y_tc, pp)),
                      "macro_f1": float(f1_score(y_tc, pp, average="macro"))}
        print(f"  {mname:7s}: {res[mname]['acc']*100:.2f}%")
    results[name] = res

out = {"protocol": "train on METABRIC(+GSE21653), evaluate on TCGA (n=987); mRNA-only, "
                   f"{len(found)} common genes, per-platform rank normalization",
       "results": results,
       "note": "Tests whether additional real training samples narrow the Transformer gap."}
with open(os.path.join(RESULTS, "multicohort_cv.json"), "w") as f:
    json.dump(out, f, indent=2)

print("\n=== TCGA accuracy, single vs two cohorts ===")
for mname in ["logreg", "tf", "tf_pre"]:
    a = results["metabric_only"][mname]["acc"]; b = results["metabric+gse21653"][mname]["acc"]
    print(f"  {mname:7s}: {a*100:.2f}% -> {b*100:.2f}%  ({100*(b-a):+.2f})")
print("wrote results/route_a/multicohort_cv.json")
