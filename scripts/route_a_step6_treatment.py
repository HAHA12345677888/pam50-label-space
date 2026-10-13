"""
Step 6: outcome prediction among treated patients.

WHAT THIS IS NOT: METABRIC's CHEMOTHERAPY / HORMONE_THERAPY / RADIO_THERAPY
fields are retrospective flags that a treatment was administered. They are not
response assessments, and there is no RECIST-style response variable in this
dataset. So this step cannot predict "response to chemotherapy". What it does
predict is 5-year overall-survival outcome among patients who received a given
treatment -- i.e. prognosis within a treated subgroup.

Reported: 5-fold CV AUC on the training split, plus a direct comparison against
a PAM50-subtype-only baseline so the genomic panel has to earn its place.

Patients are matched to the npz splits by PATIENT_ID.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, stack, folds, BEST_MODS, RESULTS, SEED
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

CLIN = "/Users/liushuxing8888student.usm.my/Downloads/clinical_merged.csv"
HORIZON = 60.0   # months


def build_targets():
    """Attach treatment flags and a 5-year alive/dead endpoint to the feature rows."""
    mods, y, dur, ev, pids = pool_splits(["train", "val", "test"])
    X = stack(mods, BEST_MODS)
    clin = pd.read_csv(CLIN, low_memory=False)
    clin = clin.drop_duplicates(subset="patientId")
    keep = ["patientId", "CHEMOTHERAPY", "HORMONE_THERAPY", "RADIO_THERAPY",
            "OS_MONTHS", "OS_STATUS", "CLAUDIN_SUBTYPE"]
    cc = clin[keep].set_index("patientId")
    df = cc.reindex(pids)
    df["label"] = y
    # 5-year outcome: dead before 60 months -> 1; alive past 60 months -> 0;
    # censored or dead after 60 -> excluded (outcome unknown at the horizon).
    d = dur.astype(float)
    e = ev.astype(int)
    outcome = np.full(len(df), np.nan)
    outcome[(e == 1) & (d <= HORIZON)] = 1.0
    outcome[(d > HORIZON)] = 0.0
    df["outcome5y"] = outcome
    return X, df


def run_arm(X, df, treat_col, label="treated"):
    """CV AUC of the genomic model vs the subtype-only baseline, among treated."""
    mask_treated = df[treat_col].eq("YES").values
    mask_known = df["outcome5y"].notna().values
    m = mask_treated & mask_known
    n = int(m.sum())
    if n < 60:
        return {"n": n, "skipped": "too few patients"}
    pos = int(df["outcome5y"].values[m].sum())
    out = {"n": n, "n_events": pos, "event_rate": float(pos / n)}
    Xm = X[m]
    ym = df["outcome5y"].values[m].astype(int)

    def auc_of(est, Xa):
        s = cross_val_score(est, Xa, ym, cv=folds(), scoring="roc_auc", n_jobs=1)
        return float(s.mean()), float(s.std())

    full = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=0.01, max_iter=5000, random_state=SEED))])
    a, s = auc_of(full, Xm)
    out["auc_genomic"] = a
    out["auc_genomic_std"] = s

    # subtype-only baseline: one-hot PAM50 call
    sub = pd.get_dummies(df["CLAUDIN_SUBTYPE"].values[m]).values.astype(float)
    b = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))])
    ab, sb = auc_of(b, sub)
    out["auc_subtype_only"] = ab
    out["auc_subtype_only_std"] = sb
    out["delta_vs_subtype"] = a - ab
    return out


if __name__ == "__main__":
    X, df = build_targets()
    print(f"rows={len(df)}  outcome5y known={df['outcome5y'].notna().sum()}  "
          f"events={int(df['outcome5y'].sum(skipna=True))}")
    res = {"horizon_months": HORIZON,
           "cohort": {"n_rows": int(len(df)),
                      "n_outcome_known": int(df["outcome5y"].notna().sum()),
                      "n_events": int(df["outcome5y"].sum(skipna=True))},
           "caveat": ("Treatment fields are administered-treatment flags, not response "
                      "assessments. This is prognosis among treated patients, not "
                      "treatment-response prediction.")}
    for col in ["CHEMOTHERAPY", "HORMONE_THERAPY", "RADIO_THERAPY"]:
        r = run_arm(X, df, col)
        res[col] = r
        if "skipped" in r:
            print(f"{col}: skipped ({r['skipped']})")
        else:
            print(f"{col}: n={r['n']} events={r['n_events']}  "
                  f"AUC genomic={r['auc_genomic']:.3f}  subtype-only={r['auc_subtype_only']:.3f}  "
                  f"delta={r['delta_vs_subtype']:+.3f}")
    with open(os.path.join(RESULTS, "treatment_response.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote results/route_a/treatment_response.json")
