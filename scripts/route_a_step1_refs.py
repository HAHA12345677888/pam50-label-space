"""Step 1: reproduce reference numbers for the LogReg and Transformer variants."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import (load_split, pool_splits, stack, make_logreg, folds,
                             BEST_MODS, RESULTS, SEED, N_FOLDS)
from route_a_tf import LightTransformer, train_one, predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def oof_guide(X_tr, y_tr, X_te, seed=SEED, n_inner=5):
    """LogReg class probabilities: out-of-fold for train, full-fit for test."""
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    g_tr = np.zeros((len(y_tr), 6), dtype=np.float32)
    for a, b in inner.split(X_tr, y_tr):
        clf = make_logreg(0.01).fit(X_tr[a], y_tr[a])
        g_tr[b] = clf.predict_proba(X_tr[b])
    clf_full = make_logreg(0.01).fit(X_tr, y_tr)
    return g_tr, clf_full.predict_proba(X_te).astype(np.float32)


def run_cv(variant, epochs=80, seed=SEED, verbose=False, **kw):
    """5-fold CV. variant in {'tf','guided'}."""
    mods, y, _, _, _ = pool_splits(["train"])
    Xg = stack(mods, BEST_MODS)
    Xs = [mods[i] for i in BEST_MODS]
    guided = variant == "guided"
    accs, f1s, hists = [], [], []
    for k, (tr, te) in enumerate(folds().split(Xg, y)):
        tm = [x[tr] for x in Xs]
        vm = [x[te] for x in Xs]
        gt = gv = None
        if guided:
            gt, gv = oof_guide(Xg[tr], y[tr], Xg[te], seed=seed)
        model = LightTransformer([x.shape[1] for x in Xs], guided=guided, **kw)
        h = train_one(model, tm, y[tr], gt, val_mods=vm, val_y=y[te], guide_va=gv,
                      epochs=epochs, seed=seed, verbose=verbose)
        p = predict(model, vm, guide=gv)
        pred = p.argmax(1)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        hists.append(h)
        if verbose:
            print(f"  fold{k}: acc={accs[-1]:.4f}")
    return {"cv": float(np.mean(accs)), "std": float(np.std(accs)),
            "per_fold": [float(a) for a in accs],
            "macro_f1": float(np.mean(f1s)), "hists": hists}


if __name__ == "__main__":
    t0 = time.time()
    out = {}
    with open(os.path.join(RESULTS, "reference_check.json")) as f:
        out.update(json.load(f))

    for name, kw in [("tf_d64_l2", dict(d_model=64, n_layers=2, nhead=4, dim_ff=128, dropout=0.3)),
                     ("guided_d64_l2", dict(d_model=64, n_layers=2, nhead=4, dim_ff=128, dropout=0.3))]:
        variant = "guided" if name.startswith("guided") else "tf"
        r = run_cv(variant, epochs=80, **kw)
        r.pop("hists")
        out[name] = r
        print(f"{name}: CV={r['cv']*100:.2f}% +/- {r['std']*100:.2f}%  macroF1={r['macro_f1']:.4f}")

    out["_elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "reference_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote reference_check.json  ({out['_elapsed_s']}s)")
