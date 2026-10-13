"""
Reviewer response: does the label-space finding generalise beyond claudin-low?

The main simulation always removes the RAREST classes, and only removes them
completely. The reviewer asks for more general absence patterns. We add two axes:

  Axis 1 - WHO is absent. Rather than always the rarest, we sweep every single
           class and every pair, so absence of a *common* or *semantically
           confusable* class (Luminal A/B) is covered, not only the rarest.
           METABRIC source priors: LumA .357 LumB .239 Her2 .125 CL .095
                                   Basal .104 Normal .079

  Axis 2 - HOW it is absent. Three annotation patterns seen in practice:
           drop   - the class is not emitted at all (target prevalence 0)
           merge  - the class is folded into a named neighbour's label
                    (claudin-low called Basal-like; Luminal B called Luminal A)
           rare   - the class survives but is heavily under-ascertained
                    (prevalence scaled to the tau_ratio boundary, 10% of source)

Reported per cell: naive accuracy, oracle-mask accuracy, EM-detected accuracy,
and the detection F1 the label-free procedure achieves.

Writes results/route_a/response_p2b_absence_modes.json
"""
import json, os, sys, itertools, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_step5_tcga import rank_norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

K = 6
NAMES = ["LumA", "LumB", "Her2", "CL", "Basal", "Normal"]
TAU_ABS, TAU_RATIO = 0.02, 0.10
N_TARGET, N_SEEDS = 1000, 10


def em_prior(P, prior_src, allowed, n_iter=1000, tol=1e-10):
    w = np.ones(P.shape[1])
    w[~allowed] = 0.0
    pt = np.zeros(P.shape[1])
    for _ in range(n_iter):
        Q = P * w
        s = Q.sum(1, keepdims=True)
        s[s == 0] = 1.0
        Q = Q / s
        pt = Q.mean(0)
        wn = np.where(prior_src > 0, pt / np.maximum(prior_src, 1e-12), 0.0)
        wn[~allowed] = 0.0
        if np.max(np.abs(wn - w)) < tol:
            w = wn
            break
        w = wn
    return w, pt


def apply_mask(P, prior_src, allowed):
    w, pt = em_prior(P.copy(), prior_src, allowed)
    Q = P * w
    s = Q.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return (Q / s).argmax(1), pt


def detect_absent(P, prior_src):
    _, pt = em_prior(P.copy(), prior_src, np.ones(K, bool))
    return (pt < TAU_ABS) & (pt < TAU_RATIO * prior_src), pt


def run_cell(Xpool, ypool, lr, sc, src_prior, absent, target_prior, seed):
    """Draw a target with the given prevalence vector; score three estimators."""
    rng = np.random.RandomState(seed)
    p = target_prior[ypool].astype(float)
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    sel = rng.choice(len(ypool), size=N_TARGET, replace=True, p=p)
    Xt, yt = Xpool[sel], ypool[sel]
    P = lr.predict_proba(sc.transform(Xt))
    base = P.argmax(1)

    allowed_o = ~absent
    pred_o = apply_mask(P.copy(), src_prior, allowed_o)[0] if allowed_o.sum() >= 2 else base
    det, _ = detect_absent(P.copy(), src_prior)
    allowed_d = ~det
    pred_d = apply_mask(P.copy(), src_prior, allowed_d)[0] if allowed_d.sum() >= 2 else base

    tp = int((det & absent).sum())
    fp = int((det & ~absent).sum())
    fn = int((~det & absent).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    det_f1 = (2 * pr * rc / max(pr + rc, 1e-9)) if (tp + fp + fn) > 0 else None

    return {"acc_base": float(accuracy_score(yt, base)),
            "acc_oracle": float(accuracy_score(yt, pred_o)),
            "acc_detected": float(accuracy_score(yt, pred_d)),
            "det_f1": det_f1}


def summarise(rows, axis, absent_names, mass, seed0, extra=None):
    res = [run_cell(*cellargs, seed=seed0 + s) for s in range(N_SEEDS)]
    res = [r for r in res if r]
    if not res:
        return None
    f1s = [r["det_f1"] for r in res if r["det_f1"] is not None]
    row = {"axis": axis, "absent": absent_names, "absent_prior_mass": float(mass),
           "acc_base": float(np.mean([r["acc_base"] for r in res])),
           "acc_oracle": float(np.mean([r["acc_oracle"] for r in res])),
           "acc_detected": float(np.mean([r["acc_detected"] for r in res])),
           "det_f1": float(np.mean(f1s)) if f1s else None}
    if extra:
        row.update(extra)
    return row


if __name__ == "__main__":
    mods, ytr, _, _, _ = pool_splits(["train"])
    mods_va, yva, _, _, _ = pool_splits(["val"])
    mods_te, yte, _, _, _ = pool_splits(["test"])
    Xtr = rank_norm(np.nan_to_num(mods[1], nan=0.0).astype(np.float64))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    src_prior = np.bincount(ytr, minlength=K).astype(float) / len(ytr)

    Xpool = np.vstack([rank_norm(np.nan_to_num(mods_va[1], nan=0.0).astype(np.float64)),
                       rank_norm(np.nan_to_num(mods_te[1], nan=0.0).astype(np.float64))])
    ypool = np.concatenate([yva, yte])

    global cellargs
    rows = []

    # ---- Axis 1a: which single class is absent, DROP mode ----
    for k in range(K):
        absent = np.zeros(K, bool)
        absent[k] = True
        tp = src_prior.copy()
        tp[k] = 0.0
        tp /= tp.sum()
        cellargs = (Xpool, ypool, lr, sc, src_prior, absent, tp)
        r = summarise(rows, "single_drop", [NAMES[k]], src_prior[k], 2000)
        if r:
            rows.append(r)

    # ---- Axis 1b: which single class is absent, RARE mode ----
    for k in range(K):
        absent = np.zeros(K, bool)
        absent[k] = True
        tp = src_prior.copy()
        tp[k] *= TAU_RATIO
        tp /= tp.sum()
        cellargs = (Xpool, ypool, lr, sc, src_prior, absent, tp)
        r = summarise(rows, "single_rare", [NAMES[k]], src_prior[k], 3000)
        if r:
            rows.append(r)

    # ---- Axis 2: MERGE mode - one class folded into a named neighbour ----
    MERGES = [(3, 4, "CL->Basal"), (1, 0, "LumB->LumA"), (2, 1, "Her2->LumB"),
              (5, 0, "Normal->LumA"), (0, 1, "LumA->LumB")]
    for src, dst, label in MERGES:
        absent = np.zeros(K, bool)
        absent[src] = True
        tp = src_prior.copy()
        tp[dst] = tp[dst] + tp[src]
        tp[src] = 0.0
        tp /= tp.sum()
        cellargs = (Xpool, ypool, lr, sc, src_prior, absent, tp)
        r = summarise(rows, "merge", [NAMES[src]], src_prior[src], 4000,
                      extra={"merged_into": NAMES[dst], "pattern": label})
        if r:
            rows.append(r)

    # ---- Axis 3: every PAIR absent (drop) ----
    for a, b in itertools.combinations(range(K), 2):
        absent = np.zeros(K, bool)
        absent[[a, b]] = True
        tp = src_prior.copy()
        tp[[a, b]] = 0.0
        tp /= tp.sum()
        cellargs = (Xpool, ypool, lr, sc, src_prior, absent, tp)
        r = summarise(rows, "pair_drop", [NAMES[a], NAMES[b]], src_prior[a] + src_prior[b], 5000)
        if r:
            rows.append(r)

    # ---- print summary ----
    for ax in ["single_drop", "single_rare", "merge", "pair_drop"]:
        sub = [r for r in rows if r["axis"] == ax]
        if not sub:
            continue
        print("\n--- %s ---" % ax)
        for r in sub:
            tag = "+".join(r["absent"]) + ((" -> " + r["merged_into"]) if "merged_into" in r else "")
            print("  %-20s mass=%.3f  base=%.4f oracle=%.4f detected=%.4f  dF1=%s" % (
                tag, r["absent_prior_mass"], r["acc_base"], r["acc_oracle"], r["acc_detected"],
                ("%.3f" % r["det_f1"]) if r["det_f1"] is not None else " n/a"))

    json.dump({"names": NAMES, "source_prior": src_prior.tolist(),
               "n_seeds": N_SEEDS, "n_target": N_TARGET,
               "tau_abs": TAU_ABS, "tau_ratio": TAU_RATIO, "rows": rows},
              open(os.path.join(RESULTS, "response_p2b_absence_modes.json"), "w"), indent=2)
    print("\nwrote results/route_a/response_p2b_absence_modes.json")
