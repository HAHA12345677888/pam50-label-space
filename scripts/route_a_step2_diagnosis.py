"""
Step 2: is the Transformer's deficit an optimization failure or a data limit?

Two probes, both run inside the canonical 5-fold split so the numbers are
comparable to the LogReg reference:

  capacity sweep  - d_model x n_layers grid, scored by fold-test accuracy.
                    If every cell plateaus well below LogReg, the architecture
                    is not the binding constraint.
  learning curves - retrain on stratified 25/50/75/100% of each fold's training
                    portion and score the held-out fold. A curve still climbing
                    at 100% means more data would help (variance-limited); a flat
                    curve means more data of this kind would not (bias-limited).

Both write per-epoch histories so the train/val gap is visible, not just the
final score.
"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, stack, folds, BEST_MODS, RESULTS, SEED
from route_a_step1_refs import oof_guide
from route_a_tf import LightTransformer, train_one, predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

EPOCHS = 60


def _fit_eval(tm, ytr, vm, yte, gt, gv, guided, seed=SEED, **kw):
    model = LightTransformer([x.shape[1] for x in tm], guided=guided, **kw)
    h = train_one(model, tm, ytr, gt, val_mods=vm, val_y=yte, guide_va=gv,
                  epochs=EPOCHS, seed=seed)
    p = predict(model, vm, guide=gv)
    return (accuracy_score(yte, p.argmax(1)),
            f1_score(yte, p.argmax(1), average="macro"), h, p)


def capacity_point(d_model, n_layers, guided, nhead=None):
    """One (d_model, n_layers) cell, averaged over the canonical folds."""
    if nhead is None:
        nhead = max(1, min(4, d_model // 16))
    mods, y, _, _, _ = pool_splits(["train"])
    Xg = stack(mods, BEST_MODS)
    Xs = [mods[i] for i in BEST_MODS]
    accs, f1s, hist0 = [], [], None
    for tr, te in folds().split(Xg, y):
        tm, vm = [x[tr] for x in Xs], [x[te] for x in Xs]
        gt = gv = None
        if guided:
            gt, gv = oof_guide(Xg[tr], y[tr], Xg[te])
        a, f, h, _ = _fit_eval(tm, y[tr], vm, y[te], gt, gv, guided,
                               d_model=d_model, n_layers=n_layers,
                               nhead=nhead, dim_ff=d_model * 2, dropout=0.3)
        accs.append(a); f1s.append(f)
        if hist0 is None:
            hist0 = h
    return {"cv": float(np.mean(accs)), "std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "per_fold": [float(a) for a in accs],
            "hist_first_fold": hist0}


def learning_curve(fractions=(0.25, 0.5, 0.75, 1.0), d_model=64, n_layers=2):
    """Stratified subsample of each fold's training portion -> fold-test score."""
    mods, y, _, _, _ = pool_splits(["train"])
    Xg = stack(mods, BEST_MODS)
    Xs = [mods[i] for i in BEST_MODS]
    res = {}
    for frac in fractions:
        accs, f1s = [], []
        for tr, te in folds().split(Xg, y):
            if frac < 1.0:
                sub, _ = train_test_split(tr, train_size=frac, stratify=y[tr],
                                          random_state=SEED)
            else:
                sub = tr
            tm, vm = [x[sub] for x in Xs], [x[te] for x in Xs]
            a, f, _, _ = _fit_eval(tm, y[sub], vm, y[te], None, None, False,
                                   d_model=d_model, n_layers=n_layers,
                                   nhead=4, dim_ff=d_model * 2, dropout=0.3)
            accs.append(a); f1s.append(f)
        n_eff = int(round(frac * 958))
        res[str(frac)] = {"n": n_eff, "cv": float(np.mean(accs)),
                          "std": float(np.std(accs)), "macro_f1": float(np.mean(f1s))}
        print(f"  frac={frac} (n~{n_eff}): acc={res[str(frac)]['cv']*100:.2f}% "
              f"+/- {res[str(frac)]['std']*100:.2f}%")
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = {}

    print("--- capacity sweep (unguided) ---")
    sweep = {}
    for dm in [32, 64, 128, 256]:
        for nl in [1, 2, 4]:
            key = f"d{dm}_l{nl}"
            r = capacity_point(dm, nl, guided=False)
            sweep[key] = r
            print(f"  {key:10s} CV={r['cv']*100:.2f}% +/- {r['std']*100:.2f}%  "
                  f"F1={r['macro_f1']:.3f}  ({time.time()-t0:.0f}s)")
            out["capacity_sweep"] = sweep
            with open(os.path.join(RESULTS, "diagnosis.json"), "w") as f:
                json.dump(out, f, indent=2)

    print("--- learning curves (d64_l2) ---")
    out["learning_curve"] = learning_curve()
    out["_elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote diagnosis.json ({out['_elapsed_s']}s)")
