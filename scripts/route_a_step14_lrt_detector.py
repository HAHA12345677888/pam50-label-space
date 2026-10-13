"""
Step 14: generative model-selection detector for label-space mismatch.

Mechanism is deliberately different from the geometry heuristic of step 12/13.
Rather than thresholding a distance ratio, we fit an explicit mixture model and
ask, by BIC, which label space explains the unlabelled target sample best:

  p(x) = sum_{k in M} pi_k * N(x ; mu_k, Sigma)      mu_k, Sigma from SOURCE labels
  pi fitted on the TARGET by EM over the allowed set M

Candidate label spaces: all 2..6-class subsets of size 0, 1 or 2 absent classes.
The winner minimises BIC = -2 loglik + (|M|-1) log n. This is a likelihood-ratio
style test with an explicit complexity penalty, calibrated by construction rather
than by a hand-set threshold.
"""
import json, os, sys, itertools, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from route_a_step12_detector import em
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

K = 6
NPCA = 40
CANDS = [frozenset()] + [frozenset(c) for c in itertools.combinations(range(K), 1)] \
        + [frozenset(c) for c in itertools.combinations(range(K), 2)]


def fit_gaussians(Z, y):
    mu, S = [], np.zeros((Z.shape[1], Z.shape[1]))
    for k in range(K):
        m = y == k
        mu.append(Z[m].mean(0))
        d = Z[m] - mu[m.sum() and k]
        S += d.T @ d
    S /= max(len(Z) - K, 1)
    S = 0.9 * S + 0.1 * np.diag(np.diag(S)) + 1e-4 * np.eye(S.shape[1])
    Si = np.linalg.inv(S)
    sign, logdet = np.linalg.slogdet(S)
    return np.array(mu), Si, logdet


def log_prob(Z, mu, Si, logdet):
    d = Z[:, None, :] - mu[None, :, :]
    maha = np.einsum("nkd,de,nke->nk", d, Si, d)
    return -0.5 * (maha + logdet + Z.shape[1] * np.log(2 * np.pi))


def bic_select(Z, LP, src_prior):
    """Return the absent-set minimising BIC, plus the per-candidate BIC table."""
    n = len(Z)
    table = {}
    for M in CANDS:
        allowed = np.ones(K, bool)
        for k in M:
            allowed[k] = False
        if allowed.sum() < 2:
            continue
        w = src_prior.copy(); w[~allowed] = 0.0
        for _ in range(500):
            Q = LP * w
            mx = Q.max(1, keepdims=True)
            Qs = np.exp(Q - mx) * w
            s = Qs.sum(1, keepdims=True); s[s == 0] = 1e-300
            R = Qs / s
            wn = np.where(src_prior > 0, R.mean(0) / np.maximum(src_prior, 1e-12), 0.0)
            wn[~allowed] = 0.0
            if wn.sum() <= 0:
                break
            wn = wn / wn.sum() * allowed.sum() / max(allowed.sum(), 1)
            if np.max(np.abs(wn - w)) < 1e-10:
                w = wn; break
            w = wn
        Q = LP * w
        mx = Q.max(1, keepdims=True)
        Qs = np.exp(Q - mx) * w
        s = Qs.sum(1, keepdims=True); s[s == 0] = 1e-300
        ll = float(np.sum(np.log(s[:, 0]) + mx[:, 0]))
        npar = int(allowed.sum()) - 1
        table[M] = ll - 0.5 * npar * np.log(n)   # BIC/2, higher is better
    best = max(table, key=table.get)
    return best, table


def main():
    mods, ytr, _, _, _ = pool_splits(["train"])
    mva, yva, _, _, _ = pool_splits(["val"])
    mte, yte, _, _, _ = pool_splits(["test"])
    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr); Ztr0 = sc.transform(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(Ztr0, ytr)
    src_prior = np.bincount(ytr, minlength=K).astype(float) / len(ytr)
    pca = PCA(NPCA, random_state=SEED).fit(Ztr0)
    Ztr = pca.transform(Ztr0)
    mu, Si, logdet = fit_gaussians(Ztr, ytr)

    Xpool = np.vstack([rank_norm(np.nan_to_num(mva[1], nan=0.0).astype(np.float64)),
                       rank_norm(np.nan_to_num(mte[1], nan=0.0).astype(np.float64))])
    ypool = np.concatenate([yva, yte]); sd = Xpool.std(0)
    order = np.argsort(np.bincount(ypool, minlength=K))

    def make(cand, cov, ps, s):
        rng = np.random.RandomState(s)
        am = np.zeros(K, bool)
        for k in cand:
            am[k] = True
        pool_idx = np.where(~am[ypool])[0]
        if len(pool_idx) < 50:
            return None
        p = None
        if ps:
            w = rng.dirichlet(np.ones(K)); w[am] = 0.0
            if w.sum() == 0:
                return None
            w /= w.sum(); p = w[ypool[pool_idx]]; p /= p.sum()
        sel = rng.choice(pool_idx, size=1000, replace=True, p=p)
        Xt = Xpool[sel]
        if cov > 0:
            Xt = rank_norm(Xt * sd[None, :] + cov * sd[None, :] * rng.randn(len(sel), Xpool.shape[1]))
        return Xt, ypool[sel], am

    # ---- null behaviour: distribution of the selected absent-set size when nothing is absent ----
    null_sizes = []
    for s in range(3000, 3030):
        b = make(frozenset(), 0.4, 0, s)
        if b is None:
            continue
        Xt, yt, _ = b
        Zt = pca.transform(sc.transform(Xt))
        LP = log_prob(Zt, mu, Si, logdet)
        best, _ = bic_select(Zt, LP, src_prior)
        null_sizes.append(len(best))
    fp_rate = float(np.mean([s > 0 for s in null_sizes]))
    print(f"null: selected a non-empty absent set in {fp_rate*100:.0f}% of {len(null_sizes)} runs "
          f"(sizes {np.bincount(null_sizes).tolist()})")

    SETS = {"none": frozenset(), "claudin-low": frozenset([3]),
            "claudinlow+Normal": frozenset([3, 5]),
            "rarest": frozenset([int(order[0])]),
            "2rarest": frozenset([int(order[0]), int(order[1])])}
    rows = []
    hdr = f"{'absent set':18s}{'cov':>5s}{'ps':>4s} | {'detF1 LRT':>10s}{'EM':>6s} | {'acc base':>9s}{'LRT':>7s}{'EM':>7s}"
    print("\n--- held-out (seeds 2000-2019) ---"); print(hdr); print("-" * len(hdr))
    for aname, cand in SETS.items():
        for cov in (0.0, 0.4, 0.8):
            for ps in (0, 1):
                lf, ef, ab, al, ae = [], [], [], [], []
                for s in range(2000, 2020):
                    b = make(cand, cov, ps, s)
                    if b is None:
                        continue
                    Xt, yt, am = b
                    Zt0 = sc.transform(Xt); P = lr.predict_proba(Zt0)
                    Zt = pca.transform(Zt0)
                    LP = log_prob(Zt, mu, Si, logdet)
                    best, _ = bic_select(Zt, LP, src_prior)
                    det = np.zeros(K, bool)
                    for k in best:
                        det[k] = True
                    base = P.argmax(1)
                    allowed = ~det
                    pred = em(P.copy(), src_prior, allowed)[0] if allowed.sum() >= 2 else base
                    _, pt = em(P.copy(), src_prior, np.ones(K, bool))
                    edet = (pt < 0.02) & (pt < 0.10 * src_prior)
                    eallowed = ~edet
                    epred = em(P.copy(), src_prior, eallowed)[0] if eallowed.sum() >= 2 else base

                    def f1v(d):
                        tp = int((d & am).sum()); fp = int((d & ~am).sum()); fn = int((~d & am).sum())
                        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                        return 2 * pr * rc / max(pr + rc, 1e-9)
                    lf.append(f1v(det)); ef.append(f1v(edet))
                    ab.append(accuracy_score(yt, base)); al.append(accuracy_score(yt, pred)); ae.append(accuracy_score(yt, epred))
                if not ab:
                    continue
                row = {"absent_set": aname, "cov_shift": cov, "prior_shift": bool(ps),
                       "lrt_det_f1": float(np.mean(lf)), "em_det_f1": float(np.mean(ef)),
                       "acc_base": float(np.mean(ab)), "acc_lrt": float(np.mean(al)), "acc_em": float(np.mean(ae))}
                rows.append(row)
                print(f"{aname:18s}{cov:5.1f}{ps:4d} | {row['lrt_det_f1']:10.2f}{row['em_det_f1']:6.2f} | "
                      f"{row['acc_base']*100:8.2f}%{row['acc_lrt']*100:6.2f}%{row['acc_em']*100:6.2f}%")

    wins = sum(1 for r in rows if r["acc_lrt"] > r["acc_em"] + 1e-9)
    loses = sum(1 for r in rows if r["acc_lrt"] < r["acc_em"] - 1e-9)
    print(f"\nLRT beats EM on accuracy in {wins}/{len(rows)} cells, loses in {loses}")
    print(f"mean detF1: LRT {np.mean([r['lrt_det_f1'] for r in rows]):.3f}  EM {np.mean([r['em_det_f1'] for r in rows]):.3f}")
    print(f"verdict: {'LRT superior' if wins > loses and np.mean([r['lrt_det_f1'] for r in rows]) > np.mean([r['em_det_f1'] for r in rows]) else 'LRT does NOT beat EM'}")
    json.dump({"npca": NPCA, "null_false_positive_rate": fp_rate,
               "null_selected_sizes": np.bincount(null_sizes).tolist(),
               "cells_lrt_beats_em": wins, "cells_em_beats_lrt": loses, "rows": rows},
              open(os.path.join(RESULTS, "detector_lrt.json"), "w"), indent=2)
    print("wrote results/route_a/detector_lrt.json")


if __name__ == "__main__":
    main()
