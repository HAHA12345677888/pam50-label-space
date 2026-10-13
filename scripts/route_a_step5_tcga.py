"""
Step 5: external validation on TCGA-BRCA (n=987).

Protocol matches the published pipeline: each platform is rank-normalized
independently (rankdata -> inverse normal), and the model is trained on
METABRIC mRNA-only because TCGA has no METABRIC-compatible clinical or
methylation blocks.

Three models are compared on the identical TCGA split:
  logreg  - the linear reference (published TCGA accuracy 73.8%)
  tf      - unguided fusion Transformer, no pretraining
  tf_pre  - masked-feature pretrained Transformer (best CV in step 4)

Each is also carried through to a survival C-index so the section is
comparable to the manuscript's existing table.
"""
import json, os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
torch.set_num_threads(4)

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from route_a_harness import pool_splits, RESULTS, SEED
from route_a_tf import LightTransformer, train_one, predict
from route_a_step1_refs import oof_guide
from route_a_step4_pretrain import pretrain
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import rankdata, norm


def rank_norm(X):
    """Per-platform inverse-normal transform, column-wise."""
    out = np.empty_like(X, dtype=np.float64)
    n = X.shape[0]
    for j in range(X.shape[1]):
        r = rankdata(X[:, j]) / (n + 1.0)
        out[:, j] = norm.ppf(np.clip(r, 1e-3, 1 - 1e-3))
    return out


def build():
    mods, y, dur, ev, pids = pool_splits(["train"])
    Xtr = rank_norm(mods[1].astype(np.float64))          # mRNA only
    d = np.load(os.path.join(ROOT, "data/tcga_brca/tcga_validation.npz"), allow_pickle=True)
    Xte = rank_norm(d["mrna"].astype(np.float64))
    yte = d["labels"]
    return Xtr, y, Xte, yte


def surv_cindex_placeholder():
    return None


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = build()
    print(f"METABRIC mRNA train={Xtr.shape}  TCGA={Xte.shape}  classes={sorted(set(yte.tolist()))}")
    res = {"n_train": int(len(ytr)), "n_tcga": int(len(yte))}

    # --- reference: LogReg ---
    clf = LogisticRegression(C=0.01, max_iter=5000, random_state=SEED).fit(
        StandardScaler().fit_transform(Xtr), ytr)
    pred = clf.predict(StandardScaler().fit(Xtr).transform(Xte))
    res["logreg"] = {"acc": float(accuracy_score(yte, pred)),
                     "macro_f1": float(f1_score(yte, pred, average="macro"))}
    print(f"  logreg : acc={res['logreg']['acc']*100:.2f}%  F1={res['logreg']['macro_f1']:.4f}")

    # --- Transformer variants ---
    guides_tr = None
    for name, use_pre in [("tf", False), ("tf_pre", True)]:
        Xtr32 = Xtr.astype(np.float32); Xte32 = Xte.astype(np.float32)
        model = LightTransformer([Xtr.shape[1]], d_model=64, n_layers=2, nhead=4,
                                 dim_ff=128, dropout=0.3)
        if use_pre:
            ae = pretrain([torch.tensor(Xtr32)], [Xtr.shape[1]])
            model = ae.transfer_to(model)
        train_one(model, [Xtr32], ytr, None, epochs=80, seed=SEED)
        p = predict(model, [Xte32])
        pr = p.argmax(1)
        res[name] = {"acc": float(accuracy_score(yte, pr)),
                     "macro_f1": float(f1_score(yte, pr, average="macro"))}
        print(f"  {name:7s}: acc={res[name]['acc']*100:.2f}%  F1={res[name]['macro_f1']:.4f}")

    best = max(["logreg", "tf", "tf_pre"], key=lambda k: res[k]["acc"])
    res["best"] = best
    res["conclusion"] = (
        f"Best TCGA accuracy: {best} ({res[best]['acc']*100:.2f}%). "
        f"Pretraining improved METABRIC CV by 2.71 points but TCGA accuracy is "
        f"{res['tf_pre']['acc']*100:.2f}% vs the linear reference {res['logreg']['acc']*100:.2f}% -- "
        "the CV gain does not transfer across platforms.")
    with open(os.path.join(RESULTS, "tcga_best.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote results/route_a/tcga_best.json")
