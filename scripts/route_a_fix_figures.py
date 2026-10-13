"""
Regenerate the figure data that the manuscript's Figure 2 and Figure 3 need.

The previous versions of these panels were hand-constructed (see
paper/generate_figures.py prior to this commit: "Build a plausible matrix",
"distribute 58", and a KM curve drawn as exp(-0.003*t) + noise). This script
replaces all three with values computed from the actual data.

Produces results/route_a/figure_data.json with:

  metabric_test  - 3-modality LogReg (Clinical+mRNA+Methylation, C=0.01)
                   refit on train, evaluated on the held-out test split.
                   Confusion matrix, per-class metrics, accuracy, macro F1.
  tcga_logreg    - mRNA-only LogReg under per-platform rank normalization,
                   trained on METABRIC train, evaluated on TCGA-BRCA.
                   Confusion matrix, per-class metrics, accuracy, macro F1.
  tcga_km        - Kaplan-Meier step curves for the median-split risk groups
                   on TCGA, where the risk score is a Cox model fitted on
                   METABRIC training predictions and applied unchanged.

Both classifiers are deterministic (lbfgs, fixed seed), so this script
reproduces the numbers in best_model/evaluation_report.txt exactly.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                            accuracy_score, f1_score)
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

CLASS_NAMES = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]
MODALITY_IDX = [0, 1, 3]          # Clinical, mRNA, Methylation


def per_class(y, pred, k):
    rep = classification_report(y, pred, labels=list(range(k)),
                                target_names=CLASS_NAMES[:k],
                                output_dict=True, zero_division=0)
    return {c: {m: float(rep[c][m]) for m in ("precision", "recall", "f1-score", "support")}
            for c in CLASS_NAMES[:k]}


# ---------------------------------------------------------------- METABRIC
# 3-modality model, refit exactly as best_model/train_best.py does.
def metabric_test():
    tr = np.load(os.path.join(ROOT, "data/train.npz"), allow_pickle=True)
    te = np.load(os.path.join(ROOT, "data/test.npz"), allow_pickle=True)
    Xtr = np.hstack([tr[f"mod_{i}"] for i in MODALITY_IDX]).astype(np.float32)
    Xte = np.hstack([te[f"mod_{i}"] for i in MODALITY_IDX]).astype(np.float32)
    ytr, yte = tr["labels"], te["labels"]

    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(max_iter=4000, C=0.01).fit(scaler.transform(Xtr), ytr)
    pred = model.predict(scaler.transform(Xte))

    cm = confusion_matrix(yte, pred, labels=list(range(6)))
    out = {"n": int(len(yte)),
           "accuracy": float(accuracy_score(yte, pred)),
           "macro_f1": float(f1_score(yte, pred, average="macro")),
           "confusion": cm.tolist(),
           "per_class": per_class(yte, pred, 6),
           "source": "LogReg C=0.01 on Clinical+mRNA+Methylation (713d), refit from data/train.npz"}
    print(f"METABRIC test: n={out['n']} acc={out['accuracy']*100:.2f}% "
          f"macroF1={out['macro_f1']:.4f}")
    print("  row sums:", cm.sum(1).tolist(), " total:", int(cm.sum()))
    return out


# ------------------------------------------------------------------- TCGA
def tcga_logreg():
    mods, ytr, _, _, _ = pool_splits(["train"])
    Xtr = rank_norm(mods[1].astype(np.float64))          # mRNA only
    d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
    Xte = rank_norm(d["mrna"].astype(np.float64))
    yte = d["labels"]

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    pred = lr.predict(sc.transform(Xte))

    # TCGA carries five true subtypes; claudin-low (index 3) is absent from this
    # cohort, but the classifier can still *predict* it. The matrix therefore has
    # five true rows and all six prediction columns, so no sample is dropped.
    present = sorted(set(yte.tolist()))
    cm = np.zeros((len(present), 6), dtype=int)
    for r, t in enumerate(present):
        for c in range(6):
            cm[r, c] = int(((yte == t) & (pred == c)).sum())
    rep = classification_report(
        yte, pred, labels=present,
        target_names=[CLASS_NAMES[i] for i in present],
        output_dict=True, zero_division=0)
    pc = {CLASS_NAMES[i]: {m: float(rep[CLASS_NAMES[i]][m])
                           for m in ("precision", "recall", "f1-score", "support")}
          for i in present}

    out = {"n": int(len(yte)), "classes_present": present,
           "class_names": [CLASS_NAMES[i] for i in present],
           "accuracy": float(accuracy_score(yte, pred)),
           "macro_f1": float(f1_score(yte, pred, average="macro")),
           "confusion": cm.tolist(), "per_class": pc,
           "source": "mRNA-only LogReg C=0.01, per-platform rank normalization"}
    print(f"TCGA: n={out['n']} acc={out['accuracy']*100:.2f}% "
          f"macroF1={out['macro_f1']:.4f}  classes={[CLASS_NAMES[i] for i in present]}")
    print("  row sums:", cm.sum(1).tolist(), " total:", int(cm.sum()))
    return out


# --------------------------------------------------------------- TCGA KM
def tcga_km():
    mods, ytr, dur_tr, ev_tr, _ = pool_splits(["train"])
    Xtr = rank_norm(mods[1].astype(np.float64))
    d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
    Xte = rank_norm(d["mrna"].astype(np.float64))
    pids = d["patient_ids"]

    pc = pd.read_csv(os.path.join(ROOT, "data/tcga_brca/patient_clinical.csv"), low_memory=False)
    pc = pc.drop_duplicates(subset="patientId").set_index("patientId")
    dur = pd.to_numeric(pc["OS_MONTHS"].reindex(pids), errors="coerce").values
    ev = pc["OS_STATUS"].reindex(pids).astype(str).str.startswith("1:").values.astype(int)
    ok = ~np.isnan(dur)

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    P_tr, P_te = lr.predict_proba(sc.transform(Xtr)), lr.predict_proba(sc.transform(Xte))

    cols = [f"p{c}" for c in lr.classes_[1:]]          # LumA reference dropped
    df = pd.DataFrame(P_tr[:, 1:], columns=cols)
    df["T"], df["E"] = dur_tr, ev_tr
    df = df[~np.isnan(df["T"])]
    cph = CoxPHFitter(penalizer=0.1).fit(df, "T", "E")
    risk = cph.predict_partial_hazard(pd.DataFrame(P_te[:, 1:], columns=cols)).values

    ci = float(concordance_index(dur[ok], -risk[ok], ev[ok]))
    hi = risk[ok] > np.median(risk[ok])
    lrt = logrank_test(dur[ok][hi], dur[ok][~hi], ev[ok][hi], ev[ok][~hi])

    # Real Kaplan-Meier step curves, one fit per risk group.
    curves = {}
    for lab, mask in [("high", hi), ("low", ~hi)]:
        kmf = KaplanMeierFitter().fit(dur[ok][mask], ev[ok][mask])
        curves[lab] = {"timeline": [float(t) for t in kmf.timeline],
                       "survival": [float(s) for s in kmf.survival_function_.iloc[:, 0]],
                       "n": int(mask.sum()), "events": int(ev[ok][mask].sum())}

    out = {"n": int(ok.sum()), "n_events": int(ev[ok].sum()),
           "c_index": ci, "logrank_p": float(lrt.p_value),
           "median_risk": float(np.median(risk[ok])),
           "curves": curves,
           "source": "CoxPHFitter(penalizer=0.1) fitted on METABRIC train predictions, "
                     "applied to TCGA; median split of partial hazard"}
    print(f"TCGA KM: n={out['n']} events={out['n_events']} C-index={ci:.4f} "
          f"log-rank p={lrt.p_value:.4f}")
    print(f"  high-risk n={curves['high']['n']} events={curves['high']['events']} | "
          f"low-risk n={curves['low']['n']} events={curves['low']['events']}")
    return out


if __name__ == "__main__":
    data = {"metabric_test": metabric_test(),
            "tcga_logreg": tcga_logreg(),
            "tcga_km": tcga_km()}
    path = os.path.join(RESULTS, "figure_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {path}")
