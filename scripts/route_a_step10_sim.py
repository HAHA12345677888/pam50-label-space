"""
Step 10: controlled simulation characterising label-space mismatch correction.

Factorial design, semi-synthetic on real METABRIC features:
  factor A  label-space mismatch: |absent classes| in {0, 1, 2}
  factor B  residual covariate shift: noise sd as fraction of feature sd, in {0.0, 0.4, 0.8}
  factor C  prior shift: target class priors perturbed (Dirichlet) or not

For each cell we report:
  acc_base      source model, full label space (the "naive" protocol)
  acc_oracle    correction using the TRUE absent set
  acc_detected  correction using the EM-detected absent set (label-free)
  det_F1        detection quality vs the true absent set
  sh_agree      split-half prior agreement (label-free validity statistic)
  gain          acc_detected - acc_base
"""
import json, os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

K = 6
TAU_ABS = 0.02          # EM prior below this -> candidate absent
TAU_RATIO = 0.10        # and below this fraction of the source prior


def panel_genes():
    for c in glob.glob("feature_names.json") + glob.glob("data/*.json"):
        j = json.load(open(c))
        if "mrna_genes" in j:
            return [g.strip() for g in j["mrna_genes"]]
    raise RuntimeError


def em_prior(P, prior_src, allowed, n_iter=1000, tol=1e-10):
    """MLLS / Saerens EM: estimate target priors from unlabelled predictions."""
    w = np.ones(P.shape[1]); w[~allowed] = 0.0
    pt = np.zeros(P.shape[1])
    for _ in range(n_iter):
        Q = P * w
        s = Q.sum(1, keepdims=True); s[s == 0] = 1.0; Q = Q / s
        pt = Q.mean(0)
        wn = np.where(prior_src > 0, pt / np.maximum(prior_src, 1e-12), 0.0)
        wn[~allowed] = 0.0
        if np.max(np.abs(wn - w)) < tol:
            w = wn; break
        w = wn
    return w, pt


def apply_mask(P, prior_src, allowed):
    w, pt = em_prior(P.copy(), prior_src, allowed)
    Q = P * w; s = Q.sum(1, keepdims=True); s[s == 0] = 1.0
    return (Q / s).argmax(1), pt


def detect_absent(P, prior_src):
    """Label-free: run EM on the FULL label space, flag classes driven to ~0."""
    _, pt = em_prior(P.copy(), prior_src, np.ones(K, bool))
    absent = (pt < TAU_ABS) & (pt < TAU_RATIO * prior_src)
    return absent, pt


def main():
    genes = panel_genes()
    mods, ytr, _, _, _ = pool_splits(["train"])
    mods_va, yva, _, _, _ = pool_splits(["val"])
    mods_te, yte, _, _, _ = pool_splits(["test"])
    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    src_prior = np.bincount(ytr, minlength=K).astype(float) / len(ytr)

    # held-out pool the synthetic target is drawn from
    Xpool = np.vstack([rank_norm(np.nan_to_num(mods_va[1], nan=0.0).astype(np.float64)),
                       rank_norm(np.nan_to_num(mods_te[1], nan=0.0).astype(np.float64))])
    ypool = np.concatenate([yva, yte])
    order = np.argsort(np.bincount(ypool, minlength=K))   # rarest first
    sd = Xpool.std(0)

    rows = []
    for n_absent in (0, 1, 2):
        for cov in (0.0, 0.4, 0.8):
            for prior_shift in (False, True):
                accs = {"base": [], "oracle": [], "detected": []}
                f1s, shs = [], []
                for seed in range(20):
                    rng = np.random.RandomState(1000 + seed)
                    absent = np.zeros(K, bool)
                    absent[order[:n_absent]] = True
                    pool_idx = np.where(~absent[ypool])[0]
                    if len(pool_idx) < 50:
                        continue
                    if prior_shift:
                        w = rng.dirichlet(np.ones(K)) 
                        w[absent] = 0.0
                        if w.sum() == 0: continue
                        w /= w.sum()
                        p = w[ypool[pool_idx]]; p = p / p.sum()
                    else:
                        p = None
                    sel = rng.choice(pool_idx, size=1000, replace=True, p=p)
                    Xt = Xpool[sel].copy()
                    if cov > 0:
                        xr = Xt * sd[None, :]
                        xr = xr + cov * sd[None, :] * rng.randn(*xr.shape)
                        Xt = rank_norm(xr)
                    yt = ypool[sel]
                    P = lr.predict_proba(sc.transform(Xt))

                    base = P.argmax(1)
                    full = np.ones(K, bool)
                    if absent.any():
                        exp_full = np.c_[np.arange(K), absent].T   # placeholder
                    # oracle correction: mask exactly the truly absent classes
                    oracle_allowed = ~absent
                    if oracle_allowed.sum() >= 2:
                        pred_or, _ = apply_mask(P.copy(), src_prior, oracle_allowed)
                    else:
                        pred_or = base
                    det_absent, _ = detect_absent(P.copy(), src_prior)
                    det_allowed = ~det_absent
                    if det_allowed.sum() >= 2:
                        pred_det, _ = apply_mask(P.copy(), src_prior, det_allowed)
                    else:
                        pred_det = base

                    accs["base"].append(accuracy_score(yt, base))
                    accs["oracle"].append(accuracy_score(yt, pred_or))
                    accs["detected"].append(accuracy_score(yt, pred_det))
                    if absent.any():
                        tp = int((det_absent & absent).sum())
                        fp = int((det_absent & ~absent).sum())
                        fn = int((~det_absent & absent).sum())
                        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                        f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
                    # split-half prior agreement (label-free validity statistic)
                    idx = rng.permutation(len(yt)); A, B = idx[:len(yt) // 2], idx[len(yt) // 2:]
                    _, pA = em_prior(P[A].copy(), src_prior, np.ones(K, bool))
                    _, pB = em_prior(P[B].copy(), src_prior, np.ones(K, bool))
                    shs.append(float(1.0 - 0.5 * np.abs(pA - pB).sum()))
                if not accs["base"]:
                    continue
                row = {"n_absent": n_absent, "cov_shift": cov, "prior_shift": prior_shift,
                       "n_seeds": len(accs["base"]),
                       "acc_base": float(np.mean(accs["base"])),
                       "acc_oracle": float(np.mean(accs["oracle"])),
                       "acc_detected": float(np.mean(accs["detected"])),
                       "gain_base_to_detected": float(np.mean(accs["detected"]) - np.mean(accs["base"])),
                       "det_f1": float(np.mean(f1s)) if f1s else None,
                       "split_half_agreement": float(np.mean(shs))}
                rows.append(row)
                print(f"absent={n_absent} cov={cov} prior_shift={int(prior_shift)} | "
                      f"base {row['acc_base']*100:5.2f}  oracle {row['acc_oracle']*100:5.2f}  "
                      f"detected {row['acc_detected']*100:5.2f}  gain {row['gain_base_to_detected']*100:+5.2f}  "
                      f"detF1 {('%.2f'%row['det_f1']) if row['det_f1'] is not None else ' n/a'}  "
                      f"shAgree {row['split_half_agreement']:.3f}")

    json.dump({"design": {"factor_label_space": [0, 1, 2],
                          "factor_covariate_shift": [0.0, 0.4, 0.8],
                          "factor_prior_shift": [False, True],
                          "n_seeds": 20, "n_target": 1000,
                          "tau_absent": TAU_ABS, "tau_ratio": TAU_RATIO},
               "rows": rows}, open(os.path.join(RESULTS, "sim_mismatch.json"), "w"), indent=2)
    print("\nwrote results/route_a/sim_mismatch.json")


if __name__ == "__main__":
    main()
