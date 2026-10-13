"""Step 15: mRNA-only cross-platform comparator.

Writes results/route_a/mrna_only_crossplatform.json -- a single mRNA-only linear
reference (LogisticRegression C=0.01 on 412 genes, within-platform rank
normalization) evaluated on the held-out METABRIC validation and test splits and
applied unchanged to TCGA-BRCA. This is the comparator that Figure 6 plots, and
it is the "mRNA-only model" cited in the manuscript's external-validation text.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "scripts")
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def prep(m):
    return rank_norm(np.nan_to_num(m[1], nan=0.0).astype(np.float64))


mtr, ytr, _, _, _ = pool_splits(["train"])
mva, yva, _, _, _ = pool_splits(["val"])
mte, yte, _, _, _ = pool_splits(["test"])
Xtr = prep(mtr)
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)

out = {}
for name, mods, y in [("val", mva, yva), ("test", mte, yte)]:
    p = lr.predict(sc.transform(prep(mods)))
    out[name] = {"acc": float(accuracy_score(y, p)),
                 "f1": float(f1_score(y, p, average="macro")), "n": int(len(y))}

d = np.load("data/tcga_brca/tcga_validation.npz", allow_pickle=True)
Xt = rank_norm(np.nan_to_num(d["mrna"], nan=0.0).astype(np.float64))
p = lr.predict(sc.transform(Xt)); y = d["labels"]
out["tcga"] = {"acc": float(accuracy_score(y, p)),
               "f1": float(f1_score(y, p, average="macro")), "n": int(len(y))}

json.dump(out, open(os.path.join(RESULTS, "mrna_only_crossplatform.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
print("wrote results/route_a/mrna_only_crossplatform.json")
