"""
Reviewer response P2.

Three things the reviewers asked for, all reusing the frozen pipeline:

  (1) Five-class training baselines. The paper trains a six-class METABRIC model
      and reconciles its output space at evaluation time. The obvious objection is
      "why not simply train on five classes?". We build both five-class variants:
        drop   - claudin-low samples removed from training (n=867)
        merge  - claudin-low relabelled as Basal-like
      and compare them with the six-class model under the same evaluation.

  (2) Calibration metrics. EM prior correction trades macro F1 for accuracy; the
      question is whether it is only redistributing probability mass. We report
      ECE (top-1 confidence) and the multiclass Brier score for base / mask_only /
      em_pam50 on both external cohorts.

  (3) Paired significance. McNemar tests between the successive estimator stages.

Writes results/route_a/response_p2_baselines.json
"""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from route_a_step7_gse21653 import load_gse, map_probes
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import chi2

CL = 3                                   # claudin-low index in the six-class space
KTX = np.array([0, 1, 2, 4, 5])          # six-class indices present under PAM50
M6TO5 = np.array([0, 1, 2, -1, 3, 4])    # six-class -> five-class index


def panel_genes():
    for c in glob.glob(os.path.join(ROOT, "feature_names.json")) + glob.glob(os.path.join(ROOT, "data/*.json")):
        j = json.load(open(c))
        if "mrna_genes" in j:
            return [g.strip() for g in j["mrna_genes"]]
    raise RuntimeError("panel not found")


def mlls(P, prior_src, allowed=None, n_iter=1000, tol=1e-9):
    """Saerens EM prior correction; `allowed` fixes structurally impossible classes."""
    K = P.shape[1]
    prior_src = prior_src.copy()
    w = np.ones(K)
    if allowed is not None:
        w[~allowed] = 0.0
    prior_tgt = prior_src
    for _ in range(n_iter):
        Q = P * w
        s = Q.sum(1, keepdims=True); s[s == 0] = 1.0
        Q /= s
        prior_tgt = Q.mean(0)
        w_new = np.where(prior_src > 0, prior_tgt / np.maximum(prior_src, 1e-12), 0.0)
        if allowed is not None:
            w_new[~allowed] = 0.0
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new; break
        w = w_new
    Q = P * w
    s = Q.sum(1, keepdims=True); s[s == 0] = 1.0
    return Q / s, prior_tgt


def ece(P, y, n_bins=15):
    """Multiclass ECE on top-1 confidence."""
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(P, y):
    Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def mcnemar(y, p1, p2):
    """McNemar with continuity correction; returns (stat, p)."""
    b = int(np.sum((p1 == y) & (p2 != y)))
    c = int(np.sum((p1 != y) & (p2 == y)))
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(stat), float(1.0 - chi2.cdf(stat, 1)), b, c


def metrics(y, pred, P):
    return {"acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "ece": ece(P, y), "brier": brier(P, y)}


# ---------------------------------------------------------------- data
genes = panel_genes()
mods, ytr6, _, _, _ = pool_splits(["train"])          # six-class METABRIC train

tc = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
X_tc_raw, y_tc6 = tc["mrna"], tc["labels"]

probes, Xg, y_gse6, _, _, _ = load_gse()               # GSE21653
pmap = map_probes(probes, Xg)
found = [g for g in genes if g in pmap]
gcols = [genes.index(g) for g in found]
X_gse_raw = Xg[[pmap[g] for g in found]].T

COHORTS = {
    "TCGA-BRCA": (X_tc_raw.astype(np.float64), y_tc6, list(range(len(genes)))),
    "GSE21653":  (X_gse_raw.astype(np.float64), y_gse6, gcols),
}

out = {"n_train_6class": int(len(ytr6)), "panel_genes": len(genes),
       "gse21653_genes_mapped": len(found), "cohorts": {}}

for cname, (Xraw, y6, cols) in COHORTS.items():
    Xsrc_full = mods[1][:, cols] if cols != list(range(len(genes))) else mods[1]
    Xs = rank_norm(np.nan_to_num(Xsrc_full, nan=0.0).astype(np.float64))
    Xt = rank_norm(np.nan_to_num(Xraw, nan=0.0).astype(np.float64))

    y5 = M6TO5[y6]                                     # target labels in five-class space
    present = np.unique(y6)
    allowed6 = np.zeros(6, bool); allowed6[present] = True

    rec = {"n": int(len(y6)), "class_counts_6": np.bincount(y6, minlength=6).tolist(),
           "genes": len(cols)}

    # --- arm A: six-class source, label-space mask (the paper's estimator) ---
    scA = StandardScaler().fit(Xs)
    lrA = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(scA.transform(Xs), ytr6)
    PA = lrA.predict_proba(scA.transform(Xt))
    priorA = np.bincount(ytr6, minlength=6).astype(float) / len(ytr6)

    pred_base = PA.argmax(1)
    Pmask = PA.copy(); Pmask[:, ~allowed6] = 0.0
    Pmask /= Pmask.sum(1, keepdims=True)
    pred_mask = Pmask.argmax(1)
    Qem, pri_em = mlls(PA.copy(), priorA, allowed=allowed6)
    pred_em = Qem.argmax(1)
    Qfull, pri_full = mlls(PA.copy(), priorA, allowed=None)

    rec["armA_sixclass_mask"] = {
        "base":    metrics(y6, pred_base, PA),
        "mask_only": metrics(y6, pred_mask, Pmask),
        "em_pam50":  metrics(y6, pred_em, Qem),
        "em_full_claudinlow_prior": float(pri_full[CL]),
        "em_pam50_prior": pri_em.tolist(),
    }

    # --- arm B: five-class source, claudin-low dropped ---
    keep = ytr6 != CL
    XsB, yB5 = Xs[keep], M6TO5[ytr6[keep]]
    scB = StandardScaler().fit(XsB)
    lrB = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(scB.transform(XsB), yB5)
    PB = lrB.predict_proba(scB.transform(Xt))
    priorB = np.bincount(yB5, minlength=5).astype(float) / len(yB5)
    pred_B = PB.argmax(1)
    QemB, priB = mlls(PB.copy(), priorB, allowed=None)
    rec["armB_fiveclass_drop"] = {
        "n_train": int(keep.sum()),
        "base": metrics(y5, pred_B, PB),
        "em":   metrics(y5, QemB.argmax(1), QemB),
    }

    # --- arm C: five-class source, claudin-low merged into Basal-like ---
    yC5 = M6TO5[ytr6].copy()
    yC5[ytr6 == CL] = M6TO5[4]
    scC = StandardScaler().fit(Xs)
    lrC = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(scC.transform(Xs), yC5)
    PC = lrC.predict_proba(scC.transform(Xt))
    priorC = np.bincount(yC5, minlength=5).astype(float) / len(yC5)
    pred_C = PC.argmax(1)
    QemC, _ = mlls(PC.copy(), priorC, allowed=None)
    rec["armC_fiveclass_merge"] = {
        "n_train": int(len(yC5)),
        "base": metrics(y5, pred_C, PC),
        "em":   metrics(y5, QemC.argmax(1), QemC),
    }

    # --- paired tests on the six-class arm, projected to the five-class space ---
    y5_from6 = M6TO5[y6]
    rec["mcnemar"] = {}
    # within the six-class arm: score every prediction in the five-class space
    for name, pa, pb in [("base_vs_mask", pred_base, pred_mask),
                         ("mask_vs_em", pred_mask, pred_em),
                         ("base_vs_em", pred_base, pred_em)]:
        stat, p, nb, nc = mcnemar(y5_from6, M6TO5[pa], M6TO5[pb])
        rec["mcnemar"][name] = {"stat": stat, "p": p, "b": nb, "c": nc}

    # cross-arm: the paper's estimator (six-class + mask + EM) versus the naive five-class models
    stat, p, nb, nc = mcnemar(y5, M6TO5[pred_em], pred_B)
    rec["mcnemar"]["sixclass_em_vs_fiveclass_drop"] = {"stat": stat, "p": p, "b": nb, "c": nc}
    stat, p, nb, nc = mcnemar(y5, M6TO5[pred_em], pred_C)
    rec["mcnemar"]["sixclass_em_vs_fiveclass_merge"] = {"stat": stat, "p": p, "b": nb, "c": nc}

    out["cohorts"][cname] = rec

    print(f"\n=== {cname} (n={len(y6)}) ===")
    A = rec["armA_sixclass_mask"]
    for nm in ["base", "mask_only", "em_pam50"]:
        m = A[nm]
        print(f"  6-class {nm:10s} acc={m['acc']*100:6.2f}%  F1={m['macro_f1']:.4f}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}")
    for arm, key in [("drop", "armB_fiveclass_drop"), ("merge", "armC_fiveclass_merge")]:
        m = rec[key]["base"]
        print(f"  5-class {arm:10s} acc={m['acc']*100:6.2f}%  F1={m['macro_f1']:.4f}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}  (n_train={rec[key]['n_train']})")
    print(f"  claudin-low EM prior (unconstrained): {A['em_full_claudinlow_prior']:.4f}")

os.makedirs(RESULTS, exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, "response_p2_baselines.json"), "w"), indent=2)
print("\nwrote results/route_a/response_p2_baselines.json")
