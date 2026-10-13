"""
Step 3: regularization / augmentation sweep for the fusion Transformer.

Everything is selected by the canonical 5-fold CV on the training split only.
The point of this step is to answer one question: can a Transformer be pushed
to the level of the LogReg reference (75.26%) using the tools that address
overfitting? Each axis is swept independently first, then the best combination
is re-scored.
"""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, stack, folds, BEST_MODS, RESULTS, SEED
from route_a_step1_refs import oof_guide
from route_a_tf import LightTransformer, train_one, predict
from sklearn.metrics import accuracy_score, f1_score

EPOCHS = 70
REFERENCE = 0.7526   # LogReg CV on the same 713-dim feature set


def cv_score(mod_drop=0.0, wd=0.01, ls=0.0, d_model=64, n_layers=2, dropout=0.3,
             class_weight=None, epochs=EPOCHS, return_hist=False):
    mods, y, _, _, _ = pool_splits(["train"])
    Xs = [mods[i] for i in BEST_MODS]
    accs, f1s, h0 = [], [], None
    for tr, te in folds().split(stack(mods, BEST_MODS), y):
        tm, vm = [x[tr] for x in Xs], [x[te] for x in Xs]
        m = LightTransformer([x.shape[1] for x in Xs], d_model=d_model,
                             n_layers=n_layers, nhead=max(1, min(4, d_model // 16)),
                             dim_ff=d_model * 2, dropout=dropout)
        h = train_one(m, tm, y[tr], None, val_mods=vm, val_y=y[te], epochs=epochs,
                      wd=wd, modality_dropout=mod_drop, label_smoothing=ls,
                      class_weight=class_weight)
        p = predict(m, vm)
        accs.append(accuracy_score(y[te], p.argmax(1)))
        f1s.append(f1_score(y[te], p.argmax(1), average="macro"))
        if h0 is None:
            h0 = h
    out = {"cv": float(np.mean(accs)), "std": float(np.std(accs)),
           "macro_f1": float(np.mean(f1s)), "per_fold": [float(a) for a in accs]}
    if return_hist:
        out["hist_first_fold"] = h0
    return out


if __name__ == "__main__":
    t0 = time.time()
    out = {"reference_logreg_cv": REFERENCE, "epochs": EPOCHS}

    # --- single-axis sweeps -------------------------------------------------
    axes = {
        "baseline":        {},
        "mod_drop_0.1":    dict(mod_drop=0.1),
        "mod_drop_0.2":    dict(mod_drop=0.2),
        "mod_drop_0.3":    dict(mod_drop=0.3),
        "mod_drop_0.5":    dict(mod_drop=0.5),
        "wd_0.05":         dict(wd=0.05),
        "wd_0.1":          dict(wd=0.1),
        "label_smooth_0.1": dict(ls=0.1),
        "dropout_0.5":     dict(dropout=0.5),
        "d32_l1":          dict(d_model=32, n_layers=1, dropout=0.3),
        "d32_l2_moddrop":  dict(d_model=32, n_layers=2, dropout=0.3, mod_drop=0.3),
    }
    res = {}
    for name, kw in axes.items():
        r = cv_score(**kw)
        res[name] = r
        flag = "  <-- beats ref" if r["cv"] > REFERENCE else ""
        print(f"  {name:20s} CV={r['cv']*100:.2f}% +/- {r['std']*100:.2f}%  F1={r['macro_f1']:.3f}{flag}  ({time.time()-t0:.0f}s)")
        out["single_axis"] = res
        with open(os.path.join(RESULTS, "regularized_cv.json"), "w") as f:
            json.dump(out, f, indent=2)

    # --- best combination ---------------------------------------------------
    best_name = max(res, key=lambda k: res[k]["cv"])
    best_kw = axes[best_name]
    print(f"\n  best single axis: {best_name} ({res[best_name]['cv']*100:.2f}%)")
    combo_kw = dict(best_kw)
    combo_kw.setdefault("mod_drop", 0.2)
    combo_kw.setdefault("wd", 0.05)
    r = cv_score(**combo_kw, return_hist=True)
    out["best_combo"] = {"config": {k: str(v) for k, v in combo_kw.items()}, **r}
    print(f"  combo {combo_kw}: CV={r['cv']*100:.2f}% +/- {r['std']*100:.2f}%  F1={r['macro_f1']:.3f}")

    out["_elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RESULTS, "regularized_cv.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote regularized_cv.json ({out['_elapsed_s']}s)")
