"""
Step 17 — A calibrated detector for classes absent from the target cohort.

Motivation. The paper's two existing detectors (geometric coherence, and an
EM prior rule) were both reported as failing. Re-reading their results shows
two separate problems, only one of which is the detector:

  (1) Metric. The published scoring function

          f1 = tp / max(tp + 0.5*(fp + fn), 1e-9)

      returns 0.0 when a detector correctly predicts the EMPTY set on a
      cohort where nothing is absent (tp = fp = fn = 0). The standard
      empty-set convention gives 1.0. So every "none" cell was scored 0.0
      regardless of whether the detector was right or wrong, and the raw
      detected sets were never written to disk, so the two cases cannot be
      told apart after the fact.

  (2) Calibration. The EM rule used two hand-set constants

          absent  <=>  pi_k < 0.02  AND  pi_k < 0.10 * pi0_k

      that were never calibrated against a null. This step replaces them with
      a simulation-calibrated, multiplicity-controlled test.

Detector. Estimate target class priors by the Saerens EM algorithm, then test
each class against a null built by resampling SOURCE validation predictions at
the same sample size. Per-class one-sided p-values feed a Benjamini-Hochberg
procedure at alpha, so the family-wise/false-discovery behaviour is explicit
rather than implicit in two magic numbers.

Two nulls are compared:
    em_cal     null from source validation predictions, unweighted
    em_cal_iw  null from source validation predictions reweighted by an
               estimated density ratio p_target(x)/p_source(x), so the null
               adapts to the covariate shift

Baselines on identical simulated cohorts:
    em_orig    the paper's uncalibrated EM rule
    geo        geometric coherence rule with a null-calibrated threshold
    naive      "absent if never predicted"
    probfreq   "mean predicted probability below a null-calibrated threshold"

Simulation protocol matches route_a_step12/13 (mRNA-only space, n_target=1000,
seeds 2000-2019, covariate shift in {0.0, 0.4, 0.8}, optional Dirichlet prior
shift) so the numbers are directly comparable to the published table.

Output: results/route_a/detector_v2.json
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_a_harness import RESULTS, SEED, pool_splits  # noqa: E402
from route_a_step12_detector import coherence, geometry  # noqa: E402

K = 6
ALPHA = 0.05
N_TARGET = 1000
N_SEED = 20
SEED_LO = 2000
CAL_SEED = 3000
B_NULL = 200
COVS = (0.0, 0.4, 0.8)
CLASS_NAMES = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]


def rank_norm(X):
    """Column-wise rank -> normal scores. Label-free, so it is legitimate to
    apply inside the target cohort during unsupervised adaptation."""
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros_like(X)
    n = X.shape[0]
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(~np.isfinite(col)) or np.nanstd(col) == 0:
            continue
        out[:, j] = norm.ppf((rankdata(col, method="average") - 0.5) / n)
    return out


# ---------------------------------------------------------------- EM prior --
def em_prior_batch(P, pi0, n_iter=300, tol=1e-9):
    """Saerens EM class-prior estimation.

    P is (..., n, K) so an entire null ensemble can be advanced in one sweep;
    the trailing two axes are the sample and class axes. Returns (..., K).
    """
    pi = np.broadcast_to(np.asarray(pi0, float), P.shape[:-2] + (len(pi0),)).copy()
    ratio = np.maximum(pi0, 1e-12)
    for _ in range(n_iter):
        W = P * (pi / ratio)[..., None, :]
        s = W.sum(-1, keepdims=True)
        W = np.divide(W, s, out=np.full_like(W, 1.0 / len(pi0)), where=s > 0)
        new = W.mean(-2)
        tot = new.sum(-1, keepdims=True)
        new = np.divide(new, tot, out=np.full_like(new, 1.0 / len(pi0)), where=tot > 0)
        if np.max(np.abs(new - pi)) < tol:
            return new
        pi = new
    return pi


def _bh(p, alpha):
    """Benjamini-Hochberg. Returns a boolean mask of rejected (absent) classes."""
    Kk = len(p)
    order = np.argsort(p)
    ranked = p[order]
    ok = ranked <= alpha * np.arange(1, Kk + 1) / Kk
    out = np.zeros(Kk, bool)
    if ok.any():
        m = int(np.max(np.where(ok)[0]))
        out[order[: m + 1]] = True
    return out


def detect_em_cal(P_target, P_null_src, pi0, weights=None, alpha=ALPHA, B=B_NULL, rng=None):
    """Calibrated EM detector.

    P_null_src : (n_src, K) source-validation probabilities used to build the null
    weights    : optional importance weights over the source rows
    """
    rng = rng or np.random.default_rng(SEED)
    n = P_target.shape[0]
    n_src = P_null_src.shape[0]

    pi_hat = em_prior_batch(P_target[None], pi0)[0]

    w = np.ones(n_src) if weights is None else np.asarray(weights, float)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        w = np.ones(n_src)
    w = w / w.sum()
    idx = rng.choice(n_src, size=(B, n), replace=True, p=w)
    null = em_prior_batch(P_null_src[idx], pi0)          # (B, K)
    p = (null <= pi_hat[None, :]).mean(0)                # low prior -> absent
    return _bh(p, alpha), pi_hat, p


def detect_em_orig(P_target, pi0):
    """The paper's rule: two hand-set constants, no calibration."""
    pi_hat = em_prior_batch(P_target[None], pi0)[0]
    return (pi_hat < 0.02) & (pi_hat < 0.10 * pi0), pi_hat


def domain_weights(P_null_src, Zt, Zs, clip=(1.0, 99.0)):
    """Importance weights over source rows, from a source-vs-target
    discriminator. The discriminator is trained to separate the two cohorts in
    feature space, and the estimated odds p(target|x)/p(source|x) is used as the
    density ratio, so the source rows that look most like target rows dominate
    the null. Unverified in-session: no reference is asserted here.

    P_null_src is accepted for signature symmetry but the ratio is estimated in
    feature space, which is where the shift lives.
    """
    X = np.vstack([Zs, Zt])
    y = np.r_[np.zeros(len(Zs)), np.ones(len(Zt))]
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
    p = clf.predict_proba(Zs)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    w = p / (1 - p) * (len(Zs) / len(Zt))
    lo, hi = np.percentile(w, clip)
    return np.clip(w, lo, hi)


# ------------------------------------------------------------- simulation ---
def build_target(Xpool, ypool, absent, cov, prior_shift, seed, sd):
    rng = np.random.RandomState(seed)
    amask = np.zeros(K, bool)
    amask[absent] = True
    pool_idx = np.where(~amask[ypool])[0]
    if len(pool_idx) < 50:
        return None
    p = None
    if prior_shift:
        w = rng.dirichlet(np.ones(K))
        w[amask] = 0.0
        if w.sum() == 0:
            return None
        w /= w.sum()
        p = w[ypool[pool_idx]]
        p /= p.sum()
    sel = rng.choice(pool_idx, size=N_TARGET, replace=True, p=p)
    Xt = Xpool[sel].copy()
    if cov > 0:
        Xt = rank_norm(Xt * sd[None, :] + cov * sd[None, :] * rng.randn(len(sel), Xpool.shape[1]))
    return Xt, amask


def score(pred_sets, true_sets):
    """Per-class micro precision/recall/F1 plus exact-set-match.

    Empty truth with empty prediction counts as a correct (exact) match and
    contributes no class-level errors -- the standard convention the published
    f1_at did not implement.
    """
    tp = fp = fn = 0
    exact = 0
    n_null = 0
    null_fp = 0
    for pred, true in zip(pred_sets, true_sets):
        pred = np.asarray(pred, dtype=bool).reshape(-1)
        true = np.asarray(true, dtype=bool).reshape(-1)
        if np.array_equal(pred, true):
            exact += 1
        tp += int((pred & true).sum())
        fp += int((pred & ~true).sum())
        fn += int((~pred & true).sum())
        if not true.any():
            n_null += 1
            null_fp += int(bool(pred.any()))
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"micro_precision": prec, "micro_recall": rec, "micro_f1": f1,
            "exact_match_rate": exact / max(len(pred_sets), 1),
            "null_cells": n_null, "null_false_positive_cells": null_fp}


def main():
    mods, ytr, _, _, _ = pool_splits(["train"])
    mva, yva, _, _, _ = pool_splits(["val"])
    mte, yte, _, _, _ = pool_splits(["test"])

    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    pi0 = np.bincount(ytr, minlength=K).astype(float) / len(ytr)
    mu, Si, within = geometry(sc.transform(Xtr), ytr)

    Xpool = np.vstack([rank_norm(np.nan_to_num(mva[1], nan=0.0).astype(np.float64)),
                       rank_norm(np.nan_to_num(mte[1], nan=0.0).astype(np.float64))])
    ypool = np.concatenate([yva, yte])
    sd = Xpool.std(0)

    # source validation probabilities: the calibration pool for the null.
    # Mirrors the size statistics of a target cohort.
    Zva = sc.transform(rank_norm(np.nan_to_num(mva[1], nan=0.0).astype(np.float64)))
    P_src = lr.predict_proba(Zva)

    order = np.argsort(np.bincount(ypool, minlength=K))
    ABSENT_SETS = {
        "none": [],
        "claudin-low": [3],
        "Normal": [5],
        "Her2": [2],
        "claudinlow+Normal": [3, 5],
        "3rarest": [int(order[0]), int(order[1]), int(order[2])],
    }

    # ---- geometric threshold, calibrated on the null across cov levels ----
    geo_tau = {}
    for cov in COVS:
        nulls = []
        for s in range(CAL_SEED, CAL_SEED + 40):
            b = build_target(Xpool, ypool, [], cov, False, s, sd)
            if b is None:
                continue
            Zt = sc.transform(b[0])
            _, r = coherence(Zt, lr.predict_proba(Zt), mu, Si, within)
            nulls.append(np.nanmax(np.nan_to_num(r, nan=0.0)))
        geo_tau[cov] = float(np.quantile(nulls, 1 - ALPHA))

    # ---- probability-frequency threshold, calibrated the same way ----
    pf_tau = {}
    for cov in COVS:
        nulls = []
        for s in range(CAL_SEED, CAL_SEED + 40):
            b = build_target(Xpool, ypool, [], cov, False, s, sd)
            if b is None:
                continue
            nulls.append(lr.predict_proba(sc.transform(b[0])).mean(0))
        pf_tau[cov] = np.quantile(np.array(nulls), ALPHA, axis=0)

    # ---- unconditional EM null (no target dependence) ----
    em_null = {}
    for cov in COVS:
        for ps in (False, True):
            em_null[(cov, ps)] = None  # built per-cell size below

    rng = np.random.default_rng(SEED)
    null_idx = {}
    for cov in COVS:
        null_idx[cov] = rng.choice(len(P_src), size=(B_NULL, N_TARGET), replace=True)

    results = {"protocol": {
        "space": "mRNA-only (412 genes), rank->normal then StandardScaler",
        "target_n": N_TARGET, "seeds": [SEED_LO, SEED_LO + N_SEED - 1],
        "covariate_shift_levels": list(COVS), "alpha": ALPHA,
        "null_replicates": B_NULL,
        "metric": "per-class micro F1 + exact-set-match; empty truth + empty "
                  "prediction counts as correct",
    }, "geo_tau": geo_tau, "pf_tau": {str(k): v.tolist() for k, v in pf_tau.items()},
        "cells": []}

    hdr = (f"{'absent set':18}{'cov':>5}{'ps':>4} | {'em_orig':>8}{'em_cal':>8}"
           f"{'em_cal_iw':>10}{'geo':>7}{'naive':>7}{'pfreq':>7}")
    print(hdr)
    print("-" * len(hdr))

    preds = {k: {"pred": [], "true": []} for k in
             ("em_orig", "em_cal", "em_cal_iw", "geo", "naive", "pfreq")}

    for aname, abslist in ABSENT_SETS.items():
        for cov in COVS:
            for ps in (False, True):
                lo = len(preds["em_orig"]["pred"])
                for s in range(SEED_LO, SEED_LO + N_SEED):
                    b = build_target(Xpool, ypool, abslist, cov, ps, s, sd)
                    if b is None:
                        continue
                    Xt, amask = b
                    Zt = sc.transform(Xt)
                    P = lr.predict_proba(Zt)

                    # --- detectors ---
                    d_orig, _ = detect_em_orig(P, pi0)
                    d_cal, _, _ = detect_em_cal(P, P_src, pi0, None, rng=rng)
                    w = domain_weights(P_src, Zt, Zva)
                    d_iw, _, _ = detect_em_cal(P, P_src, pi0, w, rng=rng)

                    _, r = coherence(Zt, P, mu, Si, within)
                    d_geo = np.nan_to_num(r, nan=-1.0) > geo_tau[cov]

                    am = P.argmax(1)
                    d_naive = np.array([not (am == k).any() for k in range(K)])

                    d_pf = P.mean(0) < pf_tau[cov]

                    # Every detector output is coerced to a flat boolean vector
                    # of length K. Without this, a scalar or object-dtype return
                    # from any detector makes the pooled equality test in score()
                    # raise on a truth-value comparison, and the failure only
                    # happens after the whole 720-cell simulation has run.
                    for name, raw in (("em_orig", d_orig), ("em_cal", d_cal),
                                      ("em_cal_iw", d_iw), ("geo", d_geo),
                                      ("naive", d_naive), ("pfreq", d_pf)):
                        v = np.asarray(raw, dtype=bool).reshape(-1)
                        assert v.shape == (K,), (name, v.shape)
                        preds[name]["pred"].append(v)
                    truth = np.asarray(amask, dtype=bool).reshape(-1)
                    assert truth.shape == (K,), truth.shape
                    for name in preds:
                        preds[name]["true"].append(truth)

                # Score this cell on its own seeds. The pooled score below hides
                # where a detector fails, and the diagnostic that matters here --
                # whether a detector false-positives when nothing is absent --
                # depends on the covariate-shift level, so it has to be kept
                # per cell. Offsets rather than a counter, so a skipped seed
                # cannot desynchronise the slice.
                for name in preds:
                    results["cells"].append({
                        "scope": "cell", "absent_set": aname, "cov": cov,
                        "prior_shift": ps, "detector": name,
                        "n_seeds": len(preds[name]["pred"]) - lo,
                        **score(preds[name]["pred"][lo:],
                                preds[name]["true"][lo:])})

    results["pooled"] = {k: score(preds[k]["pred"], preds[k]["true"]) for k in preds}

    # Raw detected/true sets, so any scoring change can be re-evaluated without
    # re-running the 720-cell simulation.
    np.savez_compressed(
        os.path.join(RESULTS, "detector_v2_raw.npz"),
        **{f"{k}_pred": np.array(preds[k]["pred"]) for k in preds},
        **{f"{k}_true": np.array(preds[k]["true"]) for k in preds},
    )

    for k, c in results["pooled"].items():
        print(f"{k:12} F1={c['micro_f1']:.3f} "
              f"P={c['micro_precision']:.3f} R={c['micro_recall']:.3f} "
              f"exact={c['exact_match_rate']:.3f} "
              f"nullFPR={c['null_false_positive_cells']}/{c['null_cells']}", flush=True)
    print("\nper-cell null false positives (cells where nothing is absent):", flush=True)
    for c in results["cells"]:
        if c["absent_set"] == "none":
            print(f"  {c['detector']:12} cov={c['cov']:<4} ps={int(c['prior_shift'])} "
                  f"F1={c['micro_f1']:.3f} nullFPR={c['null_false_positive_cells']}"
                  f"/{c['null_cells']}", flush=True)

    json.dump(results, open(os.path.join(RESULTS, "detector_v2.json"), "w"), indent=2,
              default=float)
    print("\nwrote results/route_a/detector_v2.json")


if __name__ == "__main__":
    main()
