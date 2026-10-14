#!/usr/bin/env python3
"""
用已保存的 best_model.joblib 对新数据 (.npz) 预测乳腺癌亚型。

用法:
    python predict.py --data ../data/test.npz [--model best_model.joblib] [--out predictions.csv]
"""
import os, sys, argparse
import numpy as np
import joblib
import pandas as pd

CLASS_NAMES = ["LumA", "LumB", "Her2", "claudin-low", "Basal", "Normal"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .npz with mod_* arrays")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.joblib"))
    parser.add_argument("--out", default=None, help="Output CSV path (default: next to data file)")
    args = parser.parse_args()

    # Load model
    bundle = joblib.load(args.model)
    scaler = bundle["scaler"]
    model = bundle["model"]
    modality_idx = bundle.get("modality_idx", [0, 1, 3])
    class_names = bundle.get("class_names", CLASS_NAMES)
    print(f"[MODEL] Loaded from {args.model} (modalities: {modality_idx})")

    # Load data
    d = np.load(args.data, allow_pickle=True)
    X = np.hstack([d[f"mod_{i}"] for i in modality_idx]).astype(np.float32)
    print(f"[DATA] {args.data}: {X.shape}")

    # Predict
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)
    preds = model.predict(X_s)

    # Output
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
    df.insert(0, "predicted_class", [class_names[p] for p in preds])
    df.insert(0, "predicted_label", preds)
    if "patient_ids" in d:
        df.insert(0, "patient_id", d["patient_ids"])
    if "labels" in d:
        df.insert(1, "true_label", d["labels"])

    out = args.out or (os.path.splitext(args.data)[0] + "_predictions.csv")
    df.to_csv(out, index=False)
    print(f"\n[SAVED] {out}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
