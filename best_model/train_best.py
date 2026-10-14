#!/usr/bin/env python3
"""
Best model: Clinical + mRNA + Methylation → Logistic Regression (C=0.01)

训练并在 val / test 上评估。保存 best_model.joblib（含 scaler + model）。
用法:
    python train_best.py
"""
import os, sys, argparse
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模态选择: Clinical=0, mRNA=1, Methylation=3
MODALITY_IDX = [0, 1, 3]
MODALITY_NAMES = {0: "Clinical", 1: "mRNA", 2: "CNA", 3: "Methylation", 4: "Mutation"}
CLASS_NAMES = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = np.hstack([d[f"mod_{i}"] for i in MODALITY_IDX]).astype(np.float32)
    y = d["labels"]
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(DATA_DIR, "train.npz"))
    parser.add_argument("--val", default=os.path.join(DATA_DIR, "val.npz"))
    parser.add_argument("--test", default=os.path.join(DATA_DIR, "test.npz"))
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "best_model.joblib"))
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. Load
    Xtr, ytr = load_data(args.train)
    Xva, yva = load_data(args.val)
    Xte, yte = load_data(args.test)
    print(f"[DATA] Train {Xtr.shape} | Val {Xva.shape} | Test {Xte.shape}")
    print(f"[DATA] Modalities: {[MODALITY_NAMES[i] for i in MODALITY_IDX]}")
    print(f"[DATA] Label dist (train): {np.bincount(ytr, minlength=6)}")

    # 2. 5-fold CV on train (for honest estimate)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    pipe_cv = LogisticRegression(max_iter=4000, C=args.C)
    cv_scores = cross_val_score(pipe_cv, Xtr, ytr, cv=cv, n_jobs=1)
    print(f"\n[CV] {args.cv}-fold CV acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 3. Fit on full train
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    model = LogisticRegression(max_iter=4000, C=args.C)
    model.fit(Xtr_s, ytr)
    print(f"[MODEL] LogisticRegression C={args.C}, trained on {Xtr.shape[0]} samples")

    # 4. Evaluate
    report_lines = []
    for name, X, y in [("VAL", Xva, yva), ("TEST", Xte, yte)]:
        X_s = scaler.transform(X)
        preds = model.predict(X_s)
        acc = accuracy_score(y, preds)
        print(f"\n[{name}] Accuracy: {acc:.4f}")
        report_lines.append(f"\n{'='*60}\n{name} SET\n{'='*60}")
        report_lines.append(f"Accuracy: {acc:.4f}")
        report_lines.append(classification_report(y, preds, target_names=CLASS_NAMES, zero_division=0))
        cm = confusion_matrix(y, preds)
        report_lines.append("Confusion matrix:")
        report_lines.append(str(cm))

    # 5. Save
    joblib.dump({"scaler": scaler, "model": model,
                 "modality_idx": MODALITY_IDX,
                 "class_names": CLASS_NAMES}, args.out)
    print(f"\n[SAVED] {args.out}")

    with open(os.path.join(OUT_DIR, "evaluation_report.txt"), "w") as f:
        f.write("\n".join(report_lines))
    print(f"[SAVED] {os.path.join(OUT_DIR, 'evaluation_report.txt')}")


if __name__ == "__main__":
    main()
