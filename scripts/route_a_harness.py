"""
Route A harness: shared data loading, CV protocol, and reference checks.

Design rules enforced here:
  - One canonical loader, so every experiment sees identical arrays.
  - CV folds built once from a fixed seed and reused, so models are compared
    on the same splits.
  - No experiment reads the validation set for model selection.
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = "/Users/liushuxing8888student.usm.my/breast-cancer-multimodal-transformer"
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "route_a")

# Modality names in array order inside the npz files.
MODALITIES = ["clinical", "mrna", "cna", "methylation", "mutations"]
DIMS = [69, 412, 256, 232, 173]          # -> 1142 total
# The three-modality set that won the earlier ablation: clinical + mRNA + methylation.
BEST_MODS = [0, 1, 3]                     # -> 713 features
BEST_DIMS = [69, 412, 232]

SEED = 42
N_FOLDS = 5


def load_split(split):
    """Load one of train/val/test. Returns (mods, labels, durations, events, patient_ids)."""
    d = np.load(os.path.join(DATA, f"{split}.npz"), allow_pickle=True)
    mods = [d[f"mod_{i}"] for i in range(5)]
    return mods, d["labels"], d["durations"], d["events"], d["patient_ids"]


def pool_splits(splits):
    """Concatenate several splits into one feature stack (e.g. train+val for CV)."""
    parts = [load_split(s) for s in splits]
    mods = [np.concatenate([p[0][i] for p in parts], axis=0) for i in range(5)]
    labels = np.concatenate([p[1] for p in parts])
    durations = np.concatenate([p[2] for p in parts])
    events = np.concatenate([p[3] for p in parts])
    pids = np.concatenate([p[4] for p in parts])
    return mods, labels, durations, events, pids


def stack(mods, which=BEST_MODS):
    """Horizontal concatenation of the selected modalities."""
    return np.concatenate([mods[i] for i in which], axis=1)


def make_logreg(C=0.01):
    """The reference estimator: StandardScaler -> multinomial logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        # sklearn >= 1.7 removed multi_class; multinomial is the default for
        # multiclass targets, which is what the reference model used.
        ("clf", LogisticRegression(C=C, max_iter=5000,
                                   random_state=SEED, n_jobs=-1)),
    ])


def folds():
    """Canonical fold generator (shuffled, fixed seed)."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def cv_accuracy(X, y, estimator):
    """Return (mean, std, per-fold) accuracy for a fresh estimator."""
    scores = cross_val_score(estimator, X, y, cv=folds(), scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std()), [float(s) for s in scores]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = {}

    t0 = load_split("train")
    labels = t0[1]
    out["n_train"] = int(len(labels))
    out["label_dist_train"] = np.bincount(labels, minlength=6).tolist()
    out["majority_baseline"] = float(np.bincount(labels).max() / len(labels))

    # Which pool reproduces the published 75.3%? Test both candidate protocols.
    for pool_name, splits in [("train_only", ["train"]), ("train+val", ["train", "val"])]:
        mods, y, _, _, _ = pool_splits(splits)
        X = stack(mods, BEST_MODS)
        m, s, per = cv_accuracy(X, y, make_logreg(0.01))
        out[f"logreg_713_{pool_name}"] = {"n": int(len(y)), "cv": m, "std": s, "per_fold": per}
        print(f"[{pool_name}] n={len(y)}  CV={m*100:.2f}% +/- {s*100:.2f}%")

    # Also record all-modality LogReg for contrast.
    mods, y, _, _, _ = pool_splits(["train"])
    Xall = stack(mods, [0, 1, 2, 3, 4])
    m, s, _ = cv_accuracy(Xall, y, make_logreg(0.01))
    out["logreg_all5_train_only"] = {"cv": m, "std": s}
    print(f"[all5 train] CV={m*100:.2f}% +/- {s*100:.2f}%")

    with open(os.path.join(RESULTS, "reference_check.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/route_a/reference_check.json")


if __name__ == "__main__":
    main()
