"""Recompute TCGA time-dependent C-indices under the reproduced (fully specified) risk score.

The original analysis reported a point C-index of 0.586 with 3/5/10-year values of
0.617/0.596/0.587, but its risk-score construction was not preserved in the codebase.
This recomputes the same quantities from the reproducible pipeline
(Cox fit on METABRIC train predictions -> applied to TCGA) so the manuscript reports
one internally consistent set of numbers.
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
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

mods, ytr, dur_tr, ev_tr, _ = pool_splits(["train"])
Xtr = rank_norm(mods[1])
d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
Xte = rank_norm(d["mrna"]); yte = d["labels"]
pclin = pd.read_csv(os.path.join(ROOT, "data/tcga_brca/patient_clinical.csv"))
pclin = pclin.drop_duplicates(subset=[c for c in pclin.columns if "bcr" in c.lower()][:1] or None)
pid = d["patient_ids"]

col = [c for c in pclin.columns if "bcr" in c.lower()]
key = col[0] if col else pclin.columns[0]
lut = dict(zip(pclin[key].astype(str).str[:12], zip(pclin["OS_MONTHS"], pclin["OS_STATUS"])))
dur, ev = [], []
for s in pid:
    v = lut.get(str(s)[:12])
    dur.append(v[0] if v else np.nan); ev.append(1 if v and str(v[1]).startswith("1") else 0)
dur = np.array(dur, float); ev = np.array(ev, int)

sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
cols = [f"p{c}" for c in lr.classes_[1:]]
dftr = pd.DataFrame(lr.predict_proba(sc.transform(Xtr))[:, 1:], columns=cols)
dftr["T"] = dur_tr; dftr["E"] = ev_tr
cph = CoxPHFitter(penalizer=0.1).fit(dftr[~dftr["T"].isna()], "T", "E")
risk = cph.predict_partial_hazard(
    pd.DataFrame(lr.predict_proba(sc.transform(Xte))[:, 1:], columns=cols)).values
ok = ~np.isnan(dur)

ci = concordance_index(dur[ok], -risk[ok], ev[ok])
hi = risk[ok] > np.median(risk[ok])
p = logrank_test(dur[ok][hi], dur[ok][~hi], ev[ok][hi], ev[ok][~hi]).p_value
print(f"TCGA (reproduced): n={ok.sum()} events={ev[ok].sum()}  C-index={ci:.4f}  log-rank p={p:.4f}")

tdc = {}
tmax = dur[ok].max()
for yr in (3, 5, 10):
    t = yr * 12.0
    m = (dur[ok] <= t) & (ev[ok] == 1)
    sub = np.where(m)[0]
    # c-index over event times within the horizon, censoring later observations at the horizon
    t_ = np.minimum(dur[ok], t)
    c = concordance_index(t_, -risk[ok], ((dur[ok] <= t) & (ev[ok] == 1)).astype(int))
    tdc[f"{yr}yr"] = {"c_index": float(c), "n_events": int(m.sum())}
    print(f"  {yr:2d}-year: C-index={c:.4f}  ({m.sum()} events)")

out = {"n": int(ok.sum()), "n_events": int(ev[ok].sum()),
       "c_index": float(ci), "logrank_p": float(p), "time_dependent": tdc,
       "risk_model": "CoxPHFitter(penalizer=0.1) fit on METABRIC train predictions, applied to TCGA",
       "orig_reported": {"c_index": 0.5864, "time_dependent": {"3yr": 0.6169, "5yr": 0.5963, "10yr": 0.5872}},
       "note": "Reproduced pipeline; original construction not preserved in codebase."}
json.dump(out, open(os.path.join(RESULTS, "tcga_survival.json"), "w"), indent=2)
print("wrote results/route_a/tcga_survival.json")
