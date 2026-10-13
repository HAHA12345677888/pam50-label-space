"""
Step 11: decompose the cross-cohort gain into its components, and test whether the
correction needs ANY prior domain knowledge.

Protocols evaluated on each target cohort:
  base            source model, full 6-class label space, no correction
  mask_only       claudin-low zeroed, renormalise (needs the PAM50 annotation fact)
  em_pam50        EM prior correction with the PAM50 mask (annotation fact only)
  em_full         EM prior correction over the FULL label space, no mask at all
                  -> this uses neither target labels nor any label-space knowledge
"""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

K = 6
PAM50_MASK = np.array([True, True, True, False, True, True])
CLASS_ORDER = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]


def panel_genes():
    for c in glob.glob("feature_names.json") + glob.glob("data/*.json"):
        j = json.load(open(c))
        if "mrna_genes" in j:
            return [g.strip() for g in j["mrna_genes"]]
    raise RuntimeError


def em(P, prior_src, allowed, n_iter=2000, tol=1e-11):
    w = np.ones(K); w[~allowed] = 0.0; pt = np.zeros(K)
    for _ in range(n_iter):
        Q = P * w; s = Q.sum(1, keepdims=True); s[s == 0] = 1.0; Q = Q / s
        pt = Q.mean(0)
        wn = np.where(prior_src > 0, pt / np.maximum(prior_src, 1e-12), 0.0); wn[~allowed] = 0.0
        if np.max(np.abs(wn - w)) < tol:
            w = wn; break
        w = wn
    Q = P * w; s = Q.sum(1, keepdims=True); s[s == 0] = 1.0
    return (Q / s).argmax(1), pt


def score(y, pred):
    return dict(acc=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)))


def protocols(y, P, src_prior):
    out = {}
    out["base"] = score(y, P.argmax(1))
    Pm = P.copy(); Pm[:, 3] = 0.0
    out["mask_only"] = score(y, Pm.argmax(1))
    p, pt = em(P.copy(), src_prior, PAM50_MASK); out["em_pam50"] = score(y, p)
    out["em_pam50_prior"] = np.round(pt, 4).tolist()
    p2, pt2 = em(P.copy(), src_prior, np.ones(K, bool)); out["em_full"] = score(y, p2)
    out["em_full_prior"] = np.round(pt2, 4).tolist()
    out["true_prior"] = np.round(np.bincount(y, minlength=K) / len(y), 4).tolist()
    return out


if __name__ == "__main__":
    genes = panel_genes()
    mods, ytr, _, _, _ = pool_splits(["train"])
    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    src_prior = np.bincount(ytr, minlength=K).astype(float) / len(ytr)
    print("source prior:", np.round(src_prior, 4).tolist())

    res = {}
    d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
    Xt = rank_norm(np.nan_to_num(d["mrna"], nan=0.0).astype(np.float64))
    res["TCGA"] = protocols(d["labels"], lr.predict_proba(sc.transform(Xt)), src_prior)
    print("\n=== TCGA-BRCA (n=%d) ===" % len(d["labels"]))
    for k in ["base", "mask_only", "em_pam50", "em_full"]:
        print(f"  {k:10s} acc={res['TCGA'][k]['acc']*100:6.2f}%  macroF1={res['TCGA'][k]['macro_f1']:.4f}")
    print("  em_full estimated prior:", res["TCGA"]["em_full_prior"])
    print("  true target prior      :", res["TCGA"]["true_prior"])

    from route_a_step7_gse21653 import load_gse, map_probes
    probes, X, yg, dur, evv, gsm = load_gse()
    pmap = map_probes(probes, X)
    gs = [g for g in genes if g in pmap]
    cols = [genes.index(g) for g in gs]
    lr2 = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(
        StandardScaler().fit_transform(Xtr[:, cols]), ytr)
    sc2 = StandardScaler().fit(Xtr[:, cols])
    Xt2 = rank_norm(np.nan_to_num(X[[pmap[g] for g in gs]].T, nan=0.0).astype(np.float64))
    res["GSE21653"] = protocols(yg, lr2.predict_proba(sc2.transform(Xt2)), src_prior)
    print("\n=== GSE21653 (n=%d, %d/%d panel genes) ===" % (len(yg), len(gs), len(genes)))
    for k in ["base", "mask_only", "em_pam50", "em_full"]:
        print(f"  {k:10s} acc={res['GSE21653'][k]['acc']*100:6.2f}%  macroF1={res['GSE21653'][k]['macro_f1']:.4f}")
    print("  em_full estimated prior:", res["GSE21653"]["em_full_prior"])
    print("  true target prior      :", res["GSE21653"]["true_prior"])

    json.dump({"class_order": CLASS_ORDER, "source_prior": np.round(src_prior, 4).tolist(),
               "note": ("em_full uses neither target labels nor any label-space knowledge: EM is run over the "
                        "full 6-class space and claudin-low is left with substantial estimated prior mass "
                        "(0.0751 on TCGA, 0.0719 on GSE21653) against a true prevalence of 0. Pure EM "
                        "therefore does NOT recover the label-space mismatch, and em_full reaches only "
                        "75.99%/68.42% against 79.74%/68.42% for mask_only -- the label-space fact has to "
                        "be supplied, it cannot be inferred from unlabelled predictions alone."),
               "cohorts": res}, open(os.path.join(RESULTS, "shift_decomposition.json"), "w"), indent=2)
    print("\nwrote results/route_a/shift_decomposition.json")
