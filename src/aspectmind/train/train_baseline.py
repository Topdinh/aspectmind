import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from aspectmind.data.loader import load_config, load_split  # dùng loader bạn đã viết


ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


def labels_to_multihot(label_dict: Dict[str, str]) -> np.ndarray:
    """
    Convert labels dict (aspect -> pos/neg/neu/not_mentioned) to multi-hot vector.
    For baseline detection: 1 if aspect != 'not_mentioned', else 0.
    """
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
    # 1) Load config + load splits via your loader
    cfg = load_config()

    # NOTE: paths fixed for Dataset A
    train_path = Path("data/raw/uit_visd4sa/data/train.jsonl")
    dev_path = Path("data/raw/uit_visd4sa/data/dev.jsonl")

    train_data, _ = load_split(
        train_path,
        cfg["aspect_mapping"],
        cfg["sentiment_mapping"],
        set(cfg.get("ignored_aspects", [])),
    )
    dev_data, _ = load_split(
        dev_path,
        cfg["aspect_mapping"],
        cfg["sentiment_mapping"],
        set(cfg.get("ignored_aspects", [])),
    )

    X_train, y_train = prepare_xy(train_data)
    X_dev, y_dev = prepare_xy(dev_data)

    print("Train size:", len(X_train))
    print("Dev size:", len(X_dev))
    print("Label shape:", y_train.shape)

    # 2) TF-IDF
    # Word n-grams + char n-grams thường mạnh cho tiếng Việt (không cần word segmentation sớm)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200000,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xdv = vectorizer.transform(X_dev)

    # 3) One-vs-Rest Logistic Regression (multi-label)
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        )
    )

    print("\nTraining baseline (TF-IDF char 3-5 + OvR LogisticRegression)...")
    clf.fit(Xtr, y_train)

    # 4) Predict + metrics
    y_pred = clf.predict(Xdv)

    macro = f1_score(y_dev, y_pred, average="macro")
    micro = f1_score(y_dev, y_pred, average="micro")
    per_aspect = f1_score(y_dev, y_pred, average=None)

    print("\n=== BASELINE METRICS (DEV) ===")
    print(f"Macro-F1: {macro:.4f}")
    print(f"Micro-F1: {micro:.4f}")
    print("\nPer-aspect F1:")
    for a, f1 in zip(ASPECTS, per_aspect):
        print(f" - {a}: {f1:.4f}")

    print("\nClassification report (per label):")
    print(classification_report(y_dev, y_pred, target_names=ASPECTS, zero_division=0))

    # 5) Save artifacts (tối giản cho baseline)
    out_dir = Path("runs/baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    # save vectorizer & model via joblib
    import joblib

    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(clf, out_dir / "baseline_ovr_lr.joblib")

    metrics = {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_aspect_f1": {a: float(f1) for a, f1 in zip(ASPECTS, per_aspect)},
        "aspects": ASPECTS,
        "vectorizer": "tfidf_char_3_5",
        "model": "OneVsRest(LogisticRegression)",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved baseline artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
