import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics import f1_score, classification_report

from aspectmind.data.loader import load_config, load_split


ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


def labels_to_multihot(label_dict: Dict[str, str]) -> np.ndarray:
    y = np.zeros(len(ASPECTS), dtype=np.int64)
    for i, a in enumerate(ASPECTS):
        if label_dict.get(a, "not_mentioned") != "not_mentioned":
            y[i] = 1
    return y


def prepare_xy(split_data: List[Dict]) -> Tuple[List[str], np.ndarray]:
    texts = [ex["text"] for ex in split_data]
    Y = np.stack([labels_to_multihot(ex["labels"]) for ex in split_data], axis=0)
    return texts, Y


def main():
    # Load baseline artifacts
    baseline_dir = Path("runs/baseline")
    vec_path = baseline_dir / "tfidf_vectorizer.joblib"
    model_path = baseline_dir / "baseline_ovr_lr.joblib"

    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing baseline artifacts. Expected:\n- {vec_path}\n- {model_path}"
        )

    vectorizer = joblib.load(vec_path)
    clf = joblib.load(model_path)

    # Load config
    cfg = load_config()

    # Load TEST split
    test_path = Path("data/raw/uit_visd4sa/data/test.jsonl")
    test_data, _ = load_split(
        test_path,
        cfg["aspect_mapping"],
        cfg["sentiment_mapping"],
        set(cfg.get("ignored_aspects", [])),
    )

    X_test, y_test = prepare_xy(test_data)

    print("Test size:", len(X_test))
    print("Label shape:", y_test.shape)

    Xt = vectorizer.transform(X_test)
    y_pred = clf.predict(Xt)

    macro = f1_score(y_test, y_pred, average="macro")
    micro = f1_score(y_test, y_pred, average="micro")
    per_aspect = f1_score(y_test, y_pred, average=None)

    print("\n=== BASELINE METRICS (TEST) ===")
    print(f"Macro-F1: {macro:.4f}")
    print(f"Micro-F1: {micro:.4f}")
    print("\nPer-aspect F1:")
    for a, f1 in zip(ASPECTS, per_aspect):
        print(f" - {a}: {f1:.4f}")

    print("\nClassification report (per label):")
    print(classification_report(y_test, y_pred, target_names=ASPECTS, zero_division=0))

    # Save test metrics
    metrics = {
        "split": "test",
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_aspect_f1": {a: float(f1) for a, f1 in zip(ASPECTS, per_aspect)},
        "aspects": ASPECTS,
        "artifact_dir": str(baseline_dir.as_posix()),
    }
    (baseline_dir / "metrics_test.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nSaved test metrics to: {baseline_dir / 'metrics_test.json'}")


if __name__ == "__main__":
    main()
