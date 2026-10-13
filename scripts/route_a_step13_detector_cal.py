"""
Step 13: null-calibrated geometry detector.

Step 12 failed because tau was chosen to maximise F1 on a pooled calibration set,
which selects an over-permissive threshold (precision 0.257: it deletes real
classes, and deleting a real class is far more damaging than missing an absent
one). Here tau is instead calibrated on the NULL distribution -- the statistic
observed when no class is absent -- so the per-cohort false-positive rate is
controlled at alpha by construction.

Null calibration uses only source-labelled data resampled as a pseudo-target
with all classes present, so it needs no target labels.
"""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from route_a_step12_detector import geometry, coherence, em
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

K = 6
ALPHA = 0.05


def main():
    mods, ytr, _, _, _ = pool_splits(["train"])
    mva, yva, _, _, _ = pool_splits(["val"])
    mte, yte, _, _, _ = pool_splits(["test"])
    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(Ztr, ytr)
    src_prior = np.bincount(ytr, minlength=K).astype(float) / len(ytr)
    mu, Si, within = geometry(Ztr, ytr)

    Xpool = np.vstack([rank_norm(np.nan_to_num(mva[1], nan=0.0).astype(np.float64)),
                       rank_norm(np.nan_to_num(mte[1], nan=0.0).astype(np.float64))])
    ypool = np.concatenate([yva, yte]); sd = Xpool.std(0)
    order = np.argsort(np.bincount(ypool, minlength=K))

    # ---- null distribution of max_k r_k: pseudo-targets with every class present ----
    null_max = []
    for s in range(3000, 3040):
        rng = np.random.RandomState(s)
        sel = rng.choice(len(ypool), size=1000, replace=True)
        Xt = Xpool[sel]
        if s % 3 == 1:
            Xt = rank_norm(Xt * sd[None, :] + 0.4 * sd[None, :] * rng.randn(len(sel), Xpool.shape[1]))
        elif s % 3 == 2:
            Xt = rank_norm(Xt * sd[None, :] + 0.8 * sd[None, :] * rng.randn(len(sel), Xpool.shape[1]))
        _, r = coherence(sc.transform(Xt), lr.predict_proba(sc.transform(Xt)), mu, Si, within)
        null_max.append(np.nanmax(np.nan_to_num(r, nan=0.0)))
    tau = float(np.quantile(null_max, 1 - ALPHA))
    print(f"null-calibrated tau ({1-ALPHA:.0%} level) = {tau:.3f}   "
          f"(null max r: median {np.median(null_max):.3f}, max {np.max(null_max):.3f})")

    SETS = {"none": [], "claudin-low": [3], "claudinlow+Normal": [3, 5],
            "rarest": [int(order[0])], "2rarest": [int(order[0]), int(order[1])]}
    rows = []
    hdr = f"{'absent set':18s}{'cov':>5s}{'ps':>4s} | {'detF1 geo':>9s}{'EM':>6s} | {'acc base':>9s}{'geo':>7s}{'EM':>7s}"
    print("\n--- held-out (seeds 2000-2019) ---"); print(hdr); print("-" * len(hdr))
    for aname, abslist in SETS.items():
        for cov in (0.0, 0.4, 0.8):
            for ps in (0, 1):
                gf, ef, ab, ag, ae = [], [], [], [], []
                for s in range(2000, 2020):
                    rng = np.random.RandomState(s)
                    amask = np.zeros(K, bool); amask[abslist] = True
                    pool_idx = np.where(~amask[ypool])[0]
                    if len(pool_idx) < 50:
                        continue
                    p = None
                    if ps:
                        w = rng.dirichlet(np.ones(K)); w[amask] = 0.0
                        if w.sum() == 0:
                            continue
                        w /= w.sum(); p = w[ypool[pool_idx]]; p /= p.sum()
                    sel = rng.choice(pool_idx, size=1000, replace=True, p=p)
                    Xt = Xpool[sel]
                    if cov > 0:
                        Xt = rank_norm(Xt * sd[None, :] + cov * sd[None, :] * rng.randn(len(sel), Xpool.shape[1]))
                    yt = ypool[sel]; P = lr.predict_proba(sc.transform(Xt))
                    base = P.argmax(1)
                    _, r = coherence(sc.transform(Xt), P, mu, Si, within)
                    det = np.nan_to_num(r, nan=-1e9) > tau
                    allowed = ~det
                    pred = em(P.copy(), src_prior, allowed)[0] if allowed.sum() >= 2 else base
                    # EM comparator at its own best threshold, for fairness
                    _, pt = em(P.copy(), src_prior, np.ones(K, bool))
                    edet = (pt < 0.02) & (pt < 0.10 * src_prior)
                    eallowed = ~edet
                    epred = em(P.copy(), src_prior, eallowed)[0] if eallowed.sum() >= 2 else base

                    def f1v(d):
                        tp = int((d & amask).sum()); fp = int((d & ~amask).sum()); fn = int((~d & amask).sum())
                        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                        return 2 * pr * rc / max(pr + rc, 1e-9)
                    gf.append(f1v(det)); ef.append(f1v(edet))
                    ab.append(accuracy_score(yt, base)); ag.append(accuracy_score(yt, pred)); ae.append(accuracy_score(yt, epred))
                if not ab:
                    continue
                row = {"absent_set": aname, "cov_shift": cov, "prior_shift": bool(ps),
                       "geo_det_f1": float(np.mean(gf)), "em_det_f1": float(np.mean(ef)),
                       "acc_base": float(np.mean(ab)), "acc_geo": float(np.mean(ag)), "acc_em": float(np.mean(ae))}
                rows.append(row)
                print(f"{aname:18s}{cov:5.1f}{ps:4d} | {row['geo_det_f1']:9.2f}{row['em_det_f1']:6.2f} | "
                      f"{row['acc_base']*100:8.2f}%{row['acc_geo']*100:6.2f}%{row['acc_em']*100:6.2f}%")

    wins = sum(1 for r in rows if r["acc_geo"] > r["acc_em"] + 1e-9)
    losses = sum(1 for r in rows if r["acc_geo"] < r["acc_em"] - 1e-9)
    gains = [r["acc_geo"] - r["acc_base"] for r in rows]
    print(f"\ngeo beats EM on accuracy in {wins}/{len(rows)} cells, loses in {losses}")
    print(f"geo mean gain over base: {np.mean(gains)*100:+.2f} pts (min {np.min(gains)*100:+.2f}, max {np.max(gains)*100:+.2f})")
    print(f"mean detF1: geo {np.mean([r['geo_det_f1'] for r in rows]):.3f}  EM {np.mean([r['em_det_f1'] for r in rows]):.3f}")
    json.dump({"alpha": ALPHA, "tau": tau,
               "null_max_median": float(np.median(null_max)), "null_max_max": float(np.max(null_max)),
               "cells_geo_beats_em": wins, "cells_em_beats_geo": losses, "rows": rows},
              open(os.path.join(RESULTS, "detector_calibrated.json"), "w"), indent=2)
    print("wrote results/route_a/detector_calibrated.json")


if __name__ == "__main__":
    main()
