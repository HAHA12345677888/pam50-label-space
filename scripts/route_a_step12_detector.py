"""
Step 12: a geometry-consistency detector for label-space mismatch.

Motivation. EM/MLLS estimates target priors from unlabelled predictions, but
"class k is absent" and "the model over-predicts class k" are observationally
near-degenerate, so EM leaves substantial mass on an absent class (on TCGA it
estimates claudin-low at 0.075 against a true value of 0).

Method. For each source class k we form the class centroid mu_k and a pooled
within-class covariance Sigma in the standardised feature space the classifier
sees. For target samples the model sends to class k we compute the mean
Mahalanobis distance to mu_k, and normalise by the same quantity measured on
source class-k samples:

    s_k = E[ d(x, mu_k) | x in target, argmax = k ] / E[ d(x, mu_k) | x in source, y = k ]

The statistic used for detection is the cross-class relative score

    r_k = s_k / median_j s_j

A uniform covariate shift inflates every s_k by the same factor and therefore
cancels in r_k, whereas an absent class is an outlier. Class k is flagged absent
when r_k exceeds tau. tau is calibrated on one seed block and evaluated on a
disjoint one.
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
NAMES = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]


def panel_genes():
    for c in glob.glob("feature_names.json") + glob.glob("data/*.json"):
        j = json.load(open(c))
        if "mrna_genes" in j:
            return [g.strip() for g in j["mrna_genes"]]
    raise RuntimeError


def geometry(Xs, ys):
    """Class centroids and a shrunk pooled covariance in the given space."""
    mu = np.vstack([Xs[ys == k].mean(0) if (ys == k).any() else np.zeros(Xs.shape[1])
                    for k in range(K)])
    d = Xs - mu[ys]
    S = (d.T @ d) / max(len(Xs) - K, 1)
    alpha = 0.10
    S = (1 - alpha) * S + alpha * np.diag(np.diag(S))
    S += 1e-6 * np.eye(S.shape[0])
    Si = np.linalg.inv(S)
    within = np.array([np.sqrt(np.einsum("ij,jk,ik->i", d[ys == k], Si, d[ys == k])).mean()
                       if (ys == k).any() else np.nan for k in range(K)])
    return mu, Si, within


def coherence(Xt, Pt, mu, Si, within, thresh=1.0 / K):
    """s_k per class, and the cross-class relative score r_k."""
    s = np.full(K, np.nan)
    am = Pt.argmax(1)
    for k in range(K):
        m = (am == k) if (am == k).sum() >= 20 else (Pt[:, k] >= thresh)
        if m.sum() < 10:
            continue
        d = Xt[m] - mu[k]
        s[k] = np.sqrt(np.einsum("ij,jk,ik->i", d, Si, d)).mean() / max(within[k], 1e-9)
    med = np.nanmedian(s)
    r = s / med if med > 0 else s
    return s, r


def em(P, prior_src, allowed, n_iter=2000, tol=1e-11):
    w = np.ones(K); w[~allowed] = 0.0
    for _ in range(n_iter):
        Q = P * w; z = Q.sum(1, keepdims=True); z[z == 0] = 1.0; Q = Q / z
        pt = Q.mean(0)
        wn = np.where(prior_src > 0, pt / np.maximum(prior_src, 1e-12), 0.0); wn[~allowed] = 0.0
        if np.max(np.abs(wn - w)) < tol:
            w = wn; break
        w = wn
    Q = P * w; z = Q.sum(1, keepdims=True); z[z == 0] = 1.0
    return (Q / z).argmax(1), pt


def em_detect(P, prior_src):
    _, pt = em(P.copy(), prior_src, np.ones(K, bool))
    return (pt < 0.02) & (pt < 0.10 * prior_src)


def main():
    genes = panel_genes()
    mods, ytr, _, _, _ = pool_splits(["train"])
    mva, yva, _, _, _ = pool_splits(["val"])
    mte, yte, _, _, _ = pool_splits(["test"])
    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(Ztr, ytr)
    src_prior = np.bincount(ytr, minlength=K).astype(float) / len(ytr)
    mu, Si, within = geometry(Ztr, ytr)

    Xpool = np.vstack([rank_norm(np.nan_to_num(mva[1], nan=0.0).astype(np.float64)),
                       rank_norm(np.nan_to_num(mte[1], nan=0.0).astype(np.float64))])
    ypool = np.concatenate([yva, yte])
    order = np.argsort(np.bincount(ypool, minlength=K))
    sd = Xpool.std(0)

    ABSENT_SETS = {"none": [], "claudin-low": [3], "claudinlow+Normal": [3, 5],
                   "rarest": [int(order[0])], "2rarest": [int(order[0]), int(order[1])]}

    def build(absent, cov, prior_shift, seed):
        rng = np.random.RandomState(seed)
        amask = np.zeros(K, bool); amask[absent] = True
        pool_idx = np.where(~amask[ypool])[0]
        if len(pool_idx) < 50:
            return None
        p = None
        if prior_shift:
            w = rng.dirichlet(np.ones(K)); w[amask] = 0.0
            if w.sum() == 0:
                return None
            w /= w.sum()
            p = w[ypool[pool_idx]]; p /= p.sum()
        sel = rng.choice(pool_idx, size=1000, replace=True, p=p)
        Xt = Xpool[sel].copy()
        if cov > 0:
            Xt = rank_norm(Xt * sd[None, :] + cov * sd[None, :] * rng.randn(len(sel), Xpool.shape[1]))
        return Xt, ypool[sel], amask

    # ---- calibrate tau on seeds 1000-1019, evaluate on 2000-2019 ----
    def collect(seed_lo):
        rows = []
        for aname, abslist in ABSENT_SETS.items():
            for cov in (0.0, 0.4, 0.8):
                for ps in (0, 1):
                    for s in range(seed_lo, seed_lo + 20):
                        b = build(abslist, cov, bool(ps), s)
                        if b is None:
                            continue
                        Xt, yt, amask = b
                        Zt = sc.transform(Xt)
                        P = lr.predict_proba(Zt)
                        _, r = coherence(Zt, P, mu, Si, within)
                        rows.append({"absent": amask, "r": r, "cov": cov})
        return rows

    def f1_at(rows, tau):
        tp = fp = fn = 0
        for x in rows:
            det = np.nan_to_num(x["r"], nan=-1) > tau
            tp += int((det & x["absent"]).sum()); fp += int((det & ~x["absent"]).sum())
            fn += int((~det & x["absent"]).sum())
        f1 = tp / max(tp + 0.5 * (fp + fn), 1e-9)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        return prec, rec, f1

    print("calibrating tau on seeds 1000-1019 ...")
    cal = collect(1000)
    best = max(((t, f1_at(cal, t)) for t in np.arange(0.8, 4.01, 0.05)), key=lambda x: x[1][2])
    tau = round(float(best[0]), 2)
    print(f"  tau* = {tau}   (calibration precision={best[1][0]:.3f} recall={best[1][1]:.3f} F1={best[1][2]:.3f})")

    rows = []
    print("\n--- held-out evaluation (seeds 2000-2019) ---")
    for aname, abslist in ABSENT_SETS.items():
        for cov in (0.0, 0.4, 0.8):
            for ps in (0, 1):
                geo_f1, em_f1, det_acc, base_acc, em_acc = [], [], [], [], []
                for s in range(2000, 2020):
                    b = build(abslist, cov, bool(ps), s)
                    if b is None:
                        continue
                    Xt, yt, amask = b
                    Zt = sc.transform(Xt)
                    P = lr.predict_proba(Zt)
                    _, r = coherence(Zt, P, mu, Si, within)
                    base = P.argmax(1)
                    det = np.nan_to_num(r, nan=-1) > tau
                    allowed = ~det
                    pred = em(P.copy(), src_prior, allowed)[0] if allowed.sum() >= 2 else base
                    edet = em_detect(P.copy(), src_prior)
                    eallowed = ~edet
                    epred = em(P.copy(), src_prior, eallowed)[0] if eallowed.sum() >= 2 else base

                    def f1v(d):
                        tp = int((d & amask).sum()); fp = int((d & ~amask).sum()); fn = int((~d & amask).sum())
                        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                        return 2 * pr * rc / max(pr + rc, 1e-9)
                    geo_f1.append(f1v(det)); em_f1.append(f1v(edet))
                    det_acc.append(accuracy_score(yt, pred)); base_acc.append(accuracy_score(yt, base))
                    em_acc.append(accuracy_score(yt, epred))
                row = {"absent_set": aname, "cov_shift": cov, "prior_shift": bool(ps),
                       "geo_det_f1": float(np.mean(geo_f1)), "em_det_f1": float(np.mean(em_f1)),
                       "acc_base": float(np.mean(base_acc)), "acc_geo_corrected": float(np.mean(det_acc)),
                       "acc_em_corrected": float(np.mean(em_acc))}
                rows.append(row)
                print(f"  {aname:16s} cov={cov} ps={ps} | detF1 geo {row['geo_det_f1']:.2f} vs EM {row['em_det_f1']:.2f}"
                      f" | acc base {row['acc_base']*100:5.2f} geo {row['acc_geo_corrected']*100:5.2f}"
                      f" em {row['acc_em_corrected']*100:5.2f}")

    json.dump({"tau": tau, "calibration": {"precision": best[1][0], "recall": best[1][1], "f1": best[1][2]},
               "rows": rows}, open(os.path.join(RESULTS, "detector_sim.json"), "w"), indent=2)
    print("\nwrote results/route_a/detector_sim.json")


if __name__ == "__main__":
    main()
