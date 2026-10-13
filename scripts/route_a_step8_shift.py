"""
Step 8: is the cross-cohort accuracy drop correctable?

Hypothesis: external failure is driven by class-prior shift / label-space mismatch,
not by feature-coverage loss. Arms:
  (0) baseline argmax
  (1) label-space restriction: argmax over classes annotated in the target standard
  (2) MLLS / Saerens EM on the full 6-class space
  (3) constrained MLLS: EM restricted to the target label space
  (4) oracle: true target prior (upper bound on what prior correction can give)
"""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from route_a_step7_gse21653 import load_gse, map_probes, CLASS_ORDER
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def panel_genes():
    for c in glob.glob(os.path.join(ROOT, "feature_names.json")) + glob.glob(os.path.join(ROOT, "data/*.json")):
        j = json.load(open(c))
        if "mrna_genes" in j:
            return [g.strip() for g in j["mrna_genes"]]
    raise RuntimeError("panel not found")


def fit_source(genes, cols):
    mods, y, _, _, _ = pool_splits(["train"])
    X = mods[1][:, cols]
    Xs = rank_norm(np.nan_to_num(X, nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xs)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xs), y)
    return lr, sc, y


def mlls(P, prior_src, allowed=None, n_iter=1000, tol=1e-9):
    K = P.shape[1]
    prior_src = prior_src.copy()
    w = np.ones(K)
    if allowed is not None:
        w[~allowed] = 0.0
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
    return (Q / s).argmax(1), prior_tgt


def report(name, y, P, prior_src, present):
    K = P.shape[1]
    mask = np.zeros(K, bool); mask[present] = True
    base = P.argmax(1)
    r = {}
    r["baseline"] = accuracy_score(y, base)
    r["baseline_macroF1"] = f1_score(y, base, average="macro")
    pm = np.where(mask[None, :], P, -np.inf).argmax(1)
    r["labelspace_restrict"] = accuracy_score(y, pm)
    r["labelspace_restrict_macroF1"] = f1_score(y, pm, average="macro")
    pd_m, _ = mlls(P.copy(), prior_src)
    r["mlls_full"] = accuracy_score(y, pd_m)
    pd_c, pr_c = mlls(P.copy(), prior_src, allowed=mask)
    r["mlls_constrained"] = accuracy_score(y, pd_c)
    r["mlls_constrained_macroF1"] = f1_score(y, pd_c, average="macro")
    prior_true = np.bincount(y, minlength=K).astype(float) / len(y)
    Q = P * (prior_true / np.maximum(prior_src, 1e-12)); Q /= Q.sum(1, keepdims=True)
    r["oracle_prior"] = accuracy_score(y, Q.argmax(1))
    r["prior_source"] = prior_src.tolist(); r["prior_true"] = prior_true.tolist()
    r["prior_est_constrained"] = pr_c.tolist(); r["n"] = int(len(y))
    print(f"\n--- {name} (n={len(y)}) ---")
    print(f"  baseline                 {r['baseline']*100:6.2f}%   macroF1={r['baseline_macroF1']:.4f}")
    print(f"  label-space restriction  {r['labelspace_restrict']*100:6.2f}%   macroF1={r['labelspace_restrict_macroF1']:.4f}")
    print(f"  MLLS (full 6-class)      {r['mlls_full']*100:6.2f}%")
    print(f"  MLLS (constrained)       {r['mlls_constrained']*100:6.2f}%   macroF1={r['mlls_constrained_macroF1']:.4f}")
    print(f"  oracle prior             {r['oracle_prior']*100:6.2f}%")
    print(f"  true prior  {np.round(prior_true,3).tolist()}")
    print(f"  est  prior  {np.round(pr_c,3).tolist()}")
    return r


def run_target(lr, sc, genes, X, y, name, present):
    gs = [g for g in genes if g in X["map"]]
    cols = [genes.index(g) for g in gs]
    lr2, sc2, ytr = fit_source(genes, cols)
    prior_src = np.bincount(ytr, minlength=6).astype(float) / len(ytr)
    Xte = rank_norm(np.nan_to_num(X["mat"][[X["map"][g] for g in gs]].T, nan=0.0).astype(np.float64))
    P = lr2.predict_proba(sc2.transform(Xte))
    print(f"\n=== {name}: {len(gs)}/{len(genes)} panel genes mapped, feature space {len(cols)} ===")
    return report(name, y, P, prior_src, present), len(gs)


if __name__ == "__main__":
    genes = panel_genes()
    out = {"model": "LogReg C=0.01, 412-gene mRNA panel, within-platform rank normalization",
           "class_order": CLASS_ORDER}

    d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
    tcga_genes = list(json.load(open(os.path.join(ROOT, "data/tcga_brca/tcga_genes.json")))) \
        if os.path.exists(os.path.join(ROOT, "data/tcga_brca/tcga_genes.json")) else genes
    # TCGA matrix columns follow the same panel order used at export time
    tcols = [genes.index(g) for g in tcga_genes]
    lr2, sc2, ytr = fit_source(genes, tcols)
    prior_src = np.bincount(ytr, minlength=6).astype(float) / len(ytr)
    Xte = rank_norm(np.nan_to_num(d["mrna"], nan=0.0).astype(np.float64))
    P = lr2.predict_proba(sc2.transform(Xte))
    y = d["labels"]
    out["TCGA"] = report("TCGA-BRCA (RNA-seq, 5-class)", y, P, prior_src, np.unique(y))

    probes, X, yg, dur, evv, gsm = load_gse()
    pmap = map_probes(probes, X)
    gse = {"mat": X, "map": pmap}
    r, ng = run_target(None, None, genes, gse, yg, "GSE21653 (Affymetrix, 5-class)", np.unique(yg))
    out["GSE21653"] = r; out["GSE21653"]["genes_mapped"] = ng

    with open(os.path.join(RESULTS, "crosscohort_shift.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/route_a/crosscohort_shift.json")
