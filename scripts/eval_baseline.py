# scripts/eval_baseline.py
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score

from aspectmind.data.loader import load_all_splits
from aspectmind.inference.baseline_predictor import BaselinePredictor

ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


def labels_dict_to_vec(d: Dict[str, str]) -> np.ndarray:
    # loader.py của bạn trả {"aspect": sentiment|"not_mentioned"}.
    # aspect xuất hiện nếu != "not_mentioned"
    return np.array([1 if d.get(a, "not_mentioned") != "not_mentioned" else 0 for a in ASPECTS], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="runs/baseline")
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    args = ap.parse_args()

    data = load_all_splits()[args.split]
    predictor = BaselinePredictor(args.run_dir)

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    per_aspect_true = {a: [] for a in ASPECTS}
    per_aspect_pred = {a: [] for a in ASPECTS}

    for ex in data:
        text = ex["text"]
        labels = ex["labels"]  # dict aspect->sentiment/not_mentioned

        yt = labels_dict_to_vec(labels)
        pred = predictor.predict(text)        # dict aspect->0/1
        yp = np.array([pred[a] for a in ASPECTS], dtype=np.int64)

        y_true_all.append(yt)
        y_pred_all.append(yp)

        for i, a in enumerate(ASPECTS):
            per_aspect_true[a].append(int(yt[i]))
            per_aspect_pred[a].append(int(yp[i]))

    y_true = np.stack(y_true_all, axis=0)
    y_pred = np.stack(y_pred_all, axis=0)

    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n[EVAL:{args.split.upper()}] BASELINE run_dir={args.run_dir}")
    print(f"  macro_f1 : {macro:.4f}")
    print(f"  micro_f1 : {micro:.4f}")
    print("  per-aspect F1:")
    for a in ASPECTS:
        f1 = f1_score(per_aspect_true[a], per_aspect_pred[a], average="binary", zero_division=0)
        print(f"    - {a:11s}: {float(f1):.4f}")


if __name__ == "__main__":
    main()
