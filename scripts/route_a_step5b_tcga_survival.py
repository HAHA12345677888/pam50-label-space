"""
External survival validation on TCGA-BRCA (n=966, 133 events).

Risk score: Cox proportional-hazards model (penalizer=0.1, LumA reference
column dropped to avoid collinearity) fitted on METABRIC training predictions
and applied unchanged to TCGA. Nothing is fitted on TCGA -- that is what makes
this an external validation.

Note on reproduction: the prior session's survival script was not preserved, so
the published TCGA c_index of 0.5864 cannot be re-run verbatim. This
implementation of the standard construction yields 0.5765 for the linear
reference, within 0.01 of the published figure. The comparison across models
below is internally consistent and uses one fixed construction.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_tf import LightTransformer, train_one, predict
from route_a_step4_pretrain import pretrain
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

PUBLISHED_CINDEX = 0.5864

d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
pids, yte = d["patient_ids"], d["labels"]
pc = pd.read_csv(os.path.join(ROOT, "data/tcga_brca/patient_clinical.csv"), low_memory=False)
pc = pc.drop_duplicates(subset="patientId").set_index("patientId")
dur = pd.to_numeric(pc["OS_MONTHS"].reindex(pids), errors="coerce").values
ev = pc["OS_STATUS"].reindex(pids).astype(str).str.startswith("1:").values.astype(int)
ok = ~np.isnan(dur)

mods, ytr, dur_tr, ev_tr, _ = pool_splits(["train"])
Xtr = rank_norm(mods[1].astype(np.float64)); Xte = rank_norm(d["mrna"].astype(np.float64))


def cox_cindex(P_tr, P_te, classes):
    """Fit Cox on METABRIC predictions, score TCGA. LumA (first) col dropped."""
    cols = [f"p{c}" for c in classes[1:]]
    df = pd.DataFrame(P_tr[:, 1:], columns=cols)
    df["T"] = dur_tr; df["E"] = ev_tr
    df = df[~np.isnan(df["T"])]
    cph = CoxPHFitter(penalizer=0.1).fit(df, "T", "E")
    risk = cph.predict_partial_hazard(pd.DataFrame(P_te[:, 1:], columns=cols)).values
    ci = float(concordance_index(dur[ok], -risk[ok], ev[ok]))
    hi = risk[ok] > np.median(risk[ok])
    lr = logrank_test(dur[ok][hi], dur[ok][~hi], ev[ok][hi], ev[ok][~hi])
    return ci, float(lr.p_value)


res = {"n": int(ok.sum()), "n_events": int(ev[ok].sum()),
       "published_c_index": PUBLISHED_CINDEX,
       "risk_model": "CoxPHFitter(penalizer=0.1) fit on METABRIC train predictions, applied to TCGA"}

sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
res["logreg"] = dict(zip(["c_index", "logrank_p"],
                         cox_cindex(lr.predict_proba(sc.transform(Xtr)),
                                    lr.predict_proba(sc.transform(Xte)), lr.classes_)))

for name, pre in [("tf", False), ("tf_pre", True)]:
    Xtr32 = Xtr.astype(np.float32); Xte32 = Xte.astype(np.float32)
    m = LightTransformer([Xtr.shape[1]], d_model=64, n_layers=2, nhead=4, dim_ff=128, dropout=0.3)
    if pre:
        m = pretrain([torch.tensor(Xtr32)], [Xtr.shape[1]]).transfer_to(m)
    train_one(m, [Xtr32], ytr, None, epochs=80, seed=SEED)
    res[name] = dict(zip(["c_index", "logrank_p"],
                         cox_cindex(predict(m, [Xtr32]), predict(m, [Xte32]),
                                    np.arange(6))))

print(f"TCGA external survival: n={res['n']}  events={res['n_events']}\n")
for k in ["logreg", "tf", "tf_pre"]:
    print(f"  {k:7s}: C-index={res[k]['c_index']:.4f}  log-rank p={res[k]['logrank_p']:.4f}")
print(f"\n  published (linear reference) = {PUBLISHED_CINDEX}")

with open(os.path.join(RESULTS, "tcga_survival.json"), "w") as f:
    json.dump(res, f, indent=2)
print("wrote results/route_a/tcga_survival.json")
