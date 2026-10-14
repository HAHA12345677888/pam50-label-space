"""
Step 16 — Does the multimodal advantage survive external validation?

The published external validation was mRNA-only, so the manuscript's headline
claim (clinical + mRNA + methylation beats mRNA alone) was never tested outside
METABRIC. This step tests it on the harmonised space built by step 15, where
both cohorts carry the same three layers.

Configurations, each with identical preprocessing and the same estimator
(StandardScaler -> multinomial logistic regression, C=0.01):

    1. mrna                   412 genes
    2. mrna + clinical_h       10 harmonised clinical features
    3. mrna + methylation     232 genes
    4. clinical_h + mrna + methylation   (the harmonised analogue of the
                                          flagship 713-dimension model)

For reference the script also refits the *original* 69-dimension flagship
internally, to quantify how much of the internal score is attributable to the
METABRIC-only clinical fields (NPI, GRADE, INTCLUST_*, THREEGENE_*, COHORT_*)
that cannot exist in the target cohort.

Reported per configuration:
    CV accuracy on METABRIC train+val (5-fold, fixed seed)
    METABRIC held-out test accuracy / macro F1
    TCGA external accuracy / macro F1 -- both unrestricted (6 candidate
      labels, so predictions may fall in the absent claudin-low class) and
      restricted to TCGA's five annotated labels
    bootstrap 95% CI on the external metrics
    paired McNemar test, each multimodal config vs mRNA-only

Output: results/route_a/multimodal_external.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_a_harness import DATA, RESULTS, SEED  # noqa: E402

TCGA = os.path.join(DATA, "tcga_brca")
C = 0.01
N_BOOT = 2000
CLASS_NAMES = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]


def make_model():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])


def restricted_eval(y_true, y_pred, present):
    """Keep only rows whose true label is annotated in the target cohort, and
    map predictions falling outside the annotated set to 'unassignable'."""
    m = np.isin(y_true, present)
    yt = y_true[m]
    yp = y_pred[m].copy()
    outside = ~np.isin(yp, present)
    # Restricted operation: reassign an out-of-space prediction to the nearest
    # in-space label via the model's own probabilities is NOT what the paper
    # does; the paper's restriction removes the absent class from the candidate
    # set entirely and re-argmaxes. That is handled by the caller.
    return m, yt, yp, int(outside.sum())


def boot_ci(metric_fn, y_true, y_pred, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y_true, pred_a, pred_b):
    """Exact-ish McNemar on discordant pairs (continuity-corrected chi-square)."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum(~a_ok & b_ok))     # a wrong, b right
    n10 = int(np.sum(a_ok & ~b_ok))     # a right, b wrong
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "stat": 0.0, "p": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 2 * (1 - norm.cdf(np.sqrt(max(stat, 0))))
    return {"n01": n01, "n10": n10, "stat": float(stat), "p": float(p)}


def main():
    mh = np.load(os.path.join(DATA, "metabric_harmonized.npz"), allow_pickle=True)
    th = np.load(os.path.join(TCGA, "tcga_harmonized.npz"), allow_pickle=True)

    X = {
        "tr": {k: mh[f"train_{k}"] for k in ("clinical", "mrna", "meth")},
        "va": {k: mh[f"val_{k}"] for k in ("clinical", "mrna", "meth")},
        "te": {k: mh[f"test_{k}"] for k in ("clinical", "mrna", "meth")},
    }
    y = {"tr": mh["train_labels"], "va": mh["val_labels"], "te": mh["test_labels"]}
    Xt = {k: th[k] for k in ("clinical", "mrna", "meth")}
    yt = th["labels"]
    present = sorted(np.unique(yt).tolist())
    all_labels = list(range(6))

    # pooled train+val for CV and for the final external fit
    def pool(parts, keys):
        return np.concatenate([np.concatenate([X[p][k] for k in keys], axis=1)
                               for p in parts], axis=0)

    y_train = np.concatenate([y["tr"], y["va"]])

    CONFIGS = {
        "mrna": ["mrna"],
        "mrna+clinical_h": ["mrna", "clinical"],
        "mrna+methylation": ["mrna", "meth"],
        "clinical_h+mrna+methylation": ["clinical", "mrna", "meth"],
    }

    out = {"configs": {}, "n": {"metabric_train_val": int(len(y_train)),
                                "metabric_test": int(len(y["te"])),
                                "tcga": int(len(yt))},
           "tcga_classes_present": present, "tcga_classes_absent":
               [k for k in all_labels if k not in present],
           "estimator": f"StandardScaler -> LogisticRegression(C={C})",
           "normalisation": "within-cohort rank -> normal scores",
           "cv_protocol": "5-fold stratified on METABRIC train+val, seed 42"}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, keys in CONFIGS.items():
        Xtr = pool(["tr", "va"], keys)
        Xte = np.concatenate([X["te"][k] for k in keys], axis=1)
        Xtcga = np.concatenate([Xt[k] for k in keys], axis=1)

        scores = cross_val_score(make_model(), Xtr, y_train, cv=cv, n_jobs=1)
        model = make_model().fit(Xtr, y_train)

        te_pred = model.predict(Xte)
        te_acc = float(accuracy_score(y["te"], te_pred))
        te_f1 = float(f1_score(y["te"], te_pred, average="macro", labels=all_labels))

        P_tcga = model.predict_proba(Xtcga)
        # (a) unrestricted: argmax over all six candidate labels
        pred_unres = model.classes_[P_tcga.argmax(1)]
        acc_unres = float(accuracy_score(yt, pred_unres))
        f1_unres = float(f1_score(yt, pred_unres, average="macro", labels=all_labels))
        n_absent_pred = int((~np.isin(pred_unres, present)).sum())

        # (b) restricted: remove the absent class from the candidate set, re-argmax
        keep = np.array([i for i, c in enumerate(model.classes_) if c in present])
        pred_res = model.classes_[keep][P_tcga[:, keep].argmax(1)]
        acc_res = float(accuracy_score(yt, pred_res))
        f1_res = float(f1_score(yt, pred_res, average="macro", labels=present))

        out["configs"][name] = {
            "n_features": int(Xtr.shape[1]),
            "cv_acc_mean": float(scores.mean()), "cv_acc_sd": float(scores.std()),
            "metabric_test_acc": te_acc, "metabric_test_macro_f1": te_f1,
            "tcga_acc_unrestricted": acc_unres,
            "tcga_macro_f1_unrestricted": f1_unres,
            "tcga_predicted_into_absent_class": n_absent_pred,
            "tcga_acc_restricted": acc_res,
            "tcga_macro_f1_restricted": f1_res,
            "tcga_acc_unrestricted_ci": boot_ci(accuracy_score, yt, pred_unres),
            "tcga_acc_restricted_ci": boot_ci(accuracy_score, yt, pred_res),
            "tcga_macro_f1_restricted_ci": boot_ci(
                lambda a, b: f1_score(a, b, average="macro", labels=present), yt, pred_res),
            "per_class_recall_restricted": {
                CLASS_NAMES[c]: float((pred_res[yt == c] == c).mean())
                for c in present if (yt == c).any()},
        }
        out["configs"][name]["_pred_unres"] = pred_unres
        out["configs"][name]["_pred_res"] = pred_res
        print(f"{name:32} feats={Xtr.shape[1]:4d} CV={scores.mean():.4f} "
              f"| METtest={te_acc:.4f} | TCGA unres={acc_unres:.4f} res={acc_res:.4f}")

    # paired tests vs mRNA-only
    base_unres = out["configs"]["mrna"]["_pred_unres"]
    base_res = out["configs"]["mrna"]["_pred_res"]
    for name in CONFIGS:
        out["configs"][name]["mcnemar_vs_mrna_unrestricted"] = mcnemar(
            yt, base_unres, out["configs"][name]["_pred_unres"])
        out["configs"][name]["mcnemar_vs_mrna_restricted"] = mcnemar(
            yt, base_res, out["configs"][name]["_pred_res"])

    # ---- reference: original 69-dimension flagship, internal only ----
    d = {s: np.load(os.path.join(DATA, f"{s}.npz"), allow_pickle=True)
         for s in ("train", "val", "test")}
    Xf = np.concatenate([np.concatenate([d[s]["mod_0"], d[s]["mod_1"], d[s]["mod_3"]], axis=1)
                         for s in ("train", "val")], axis=0)
    yf = np.concatenate([d[s]["labels"] for s in ("train", "val")])
    Xf_te = np.concatenate([d["test"]["mod_0"], d["test"]["mod_1"], d["test"]["mod_3"]], axis=1)
    sc = cross_val_score(make_model(), Xf, yf, cv=cv, n_jobs=1)
    mf = make_model().fit(Xf, yf)
    pf = mf.predict(Xf_te)
    out["reference_original_flagship_69dim"] = {
        "n_features": int(Xf.shape[1]),
        "cv_acc_mean": float(sc.mean()), "cv_acc_sd": float(sc.std()),
        "metabric_test_acc": float(accuracy_score(d["test"]["labels"], pf)),
        "note": "clinical block is METABRIC-specific; has no external counterpart",
    }
    print(f"{'orig flagship (69d clinical)':32} feats={Xf.shape[1]:4d} "
          f"CV={sc.mean():.4f} | METtest={out['reference_original_flagship_69dim']['metabric_test_acc']:.4f}")

    for cfg in out["configs"].values():
        cfg.pop("_pred_unres", None)
        cfg.pop("_pred_res", None)
    json.dump(out, open(os.path.join(RESULTS, "multimodal_external.json"), "w"), indent=2)
    print("\nwrote results/route_a/multimodal_external.json")


if __name__ == "__main__":
    main()
