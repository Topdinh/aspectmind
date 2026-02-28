#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate BaselinePredictor and PhoBERTSinglePredictor on UIT-ViSD4SA jsonl.

What this script evaluates:
- Multi-label aspect detection (0/1 per aspect), NOT sentiment polarity.
- Ground-truth aspects are extracted from jsonl "labels" field (e.g., "BATTERY#POSITIVE").
- Aspects mapping is read from a dataset yaml if provided; otherwise uses a sensible default mapping.

Metrics:
- Micro Precision/Recall/F1
- Macro F1 (average of per-aspect F1)
- Label Accuracy: (TP + TN) / (TP + TN + FP + FN) across all aspect labels
- Subset Accuracy: exact match across all aspects (optional, printed)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional dependency (your repo already uses yaml in test_mapping.py)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# ----------------------------
# Defaults (safe fallback)
# ----------------------------
DEFAULT_TARGET_ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]

# Raw aspect tags in UIT-ViSD4SA → your project target aspects
DEFAULT_ASPECT_MAPPING = {
    "BATTERY": "battery",
    "CAMERA": "camera",
    "PERFORMANCE": "performance",
    "DESIGN": "design",
    "PRICE": "price",
    "SER&ACC": "service",
}

DEFAULT_IGNORED_ASPECTS = {"GENERAL"}  # common in this dataset


# ----------------------------
# Helpers
# ----------------------------
def parse_label_tag(tag: str) -> Tuple[Optional[str], Optional[str]]:
    # Example: "BATTERY#POSITIVE"
    if "#" not in tag:
        return None, None
    a, s = tag.split("#", 1)
    return a.strip(), s.strip()


def load_dataset_config(dataset_yaml: Optional[Path]):
    """
    Return (target_aspects, aspect_mapping, ignored_aspects)
    """
    if dataset_yaml is None:
        return DEFAULT_TARGET_ASPECTS, DEFAULT_ASPECT_MAPPING, DEFAULT_IGNORED_ASPECTS

    if not dataset_yaml.exists():
        print(f"[WARN] dataset yaml not found: {dataset_yaml}. Using defaults.")
        return DEFAULT_TARGET_ASPECTS, DEFAULT_ASPECT_MAPPING, DEFAULT_IGNORED_ASPECTS

    if yaml is None:
        print("[WARN] PyYAML not installed. Using defaults.")
        return DEFAULT_TARGET_ASPECTS, DEFAULT_ASPECT_MAPPING, DEFAULT_IGNORED_ASPECTS

    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))

    target_aspects = cfg.get("target_aspects", DEFAULT_TARGET_ASPECTS)
    aspect_mapping = cfg.get("aspect_mapping", DEFAULT_ASPECT_MAPPING)
    ignored_aspects = set(cfg.get("ignored_aspects", list(DEFAULT_IGNORED_ASPECTS)))

    return target_aspects, aspect_mapping, ignored_aspects


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_gold_aspects(
    obj: dict,
    target_aspects: List[str],
    aspect_mapping: Dict[str, str],
    ignored_aspects: set,
    text_field: str = "text",
    label_field: str = "labels",
) -> Dict[str, int]:
    """
    Convert labels spans into a multi-label vector for aspect presence: {aspect: 0/1}.
    If an aspect appears at least once in labels => 1 else 0.
    """
    y = {a: 0 for a in target_aspects}

    labels = obj.get(label_field, [])
    for start, end, tag in labels:
        raw_aspect, _raw_sent = parse_label_tag(tag)
        if raw_aspect is None:
            continue
        if raw_aspect in ignored_aspects:
            continue
        mapped = aspect_mapping.get(raw_aspect)
        if mapped in y:
            y[mapped] = 1

    return y


def normalize_pred_to_binary(pred: object, target_aspects: List[str]) -> Dict[str, int]:
    """
    Make predictor output robust.

    Expected (common in your tests):
    - dict aspect->0/1
    But we also handle:
    - list of aspects
    - dict aspect->bool
    - dict aspect->sentiment string (rare) -> treat presence as 1 if not 'NONE'
    """
    y = {a: 0 for a in target_aspects}

    if isinstance(pred, dict):
        for a in target_aspects:
            if a not in pred:
                continue
            v = pred[a]
            if isinstance(v, (int, np.integer)):
                y[a] = 1 if int(v) != 0 else 0
            elif isinstance(v, bool):
                y[a] = 1 if v else 0
            elif isinstance(v, str):
                # If model returns something like "POSITIVE"/"NEGATIVE"/"NEUTRAL"
                # treat it as present
                y[a] = 0 if v.strip().upper() in {"NONE", "NULL", "N/A", ""} else 1
            else:
                # fallback: truthy => present
                y[a] = 1 if v else 0
        return y

    if isinstance(pred, list):
        # list of aspects present
        for a in pred:
            if a in y:
                y[a] = 1
        return y

    # unknown type
    return y


@dataclass
class Metrics:
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    label_accuracy: float
    subset_accuracy: float
    per_aspect_f1: Dict[str, float]
    support_pos: Dict[str, int]


def compute_metrics(
    gold: List[Dict[str, int]],
    pred: List[Dict[str, int]],
    target_aspects: List[str],
) -> Metrics:
    # Flatten to arrays [N, A]
    y_true = np.array([[g[a] for a in target_aspects] for g in gold], dtype=int)
    y_pred = np.array([[p[a] for a in target_aspects] for p in pred], dtype=int)

    # Confusion components per label
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    micro_precision = tp / (tp + fp + 1e-12)
    micro_recall = tp / (tp + fn + 1e-12)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-12)

    # Per-aspect F1
    per_aspect_f1: Dict[str, float] = {}
    support_pos: Dict[str, int] = {}
    f1s = []

    for j, a in enumerate(target_aspects):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        tp_a = int(((yt == 1) & (yp == 1)).sum())
        fp_a = int(((yt == 0) & (yp == 1)).sum())
        fn_a = int(((yt == 1) & (yp == 0)).sum())

        prec_a = tp_a / (tp_a + fp_a + 1e-12)
        rec_a = tp_a / (tp_a + fn_a + 1e-12)
        f1_a = 2 * prec_a * rec_a / (prec_a + rec_a + 1e-12)

        per_aspect_f1[a] = float(f1_a)
        support_pos[a] = int(yt.sum())
        f1s.append(f1_a)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    label_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    # Subset accuracy: exact match on all aspects
    subset_accuracy = float((y_true == y_pred).all(axis=1).mean())

    return Metrics(
        micro_precision=float(micro_precision),
        micro_recall=float(micro_recall),
        micro_f1=float(micro_f1),
        macro_f1=float(macro_f1),
        label_accuracy=float(label_accuracy),
        subset_accuracy=float(subset_accuracy),
        per_aspect_f1=per_aspect_f1,
        support_pos=support_pos,
    )


def evaluate_predictor(predictor, data_path: Path, target_aspects, aspect_mapping, ignored_aspects, max_samples: Optional[int]) -> Metrics:
    gold_list: List[Dict[str, int]] = []
    pred_list: List[Dict[str, int]] = []

    for i, obj in enumerate(iter_jsonl(data_path)):
        if max_samples is not None and i >= max_samples:
            break

        text = obj.get("text", "")
        gold = extract_gold_aspects(obj, target_aspects, aspect_mapping, ignored_aspects)
        pred_raw = predictor.predict(text)
        pred_bin = normalize_pred_to_binary(pred_raw, target_aspects)

        gold_list.append(gold)
        pred_list.append(pred_bin)

    return compute_metrics(gold_list, pred_list, target_aspects)


def print_report(title: str, m: Metrics, target_aspects: List[str]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Micro Precision : {m.micro_precision:.4f}")
    print(f"Micro Recall    : {m.micro_recall:.4f}")
    print(f"Micro F1        : {m.micro_f1:.4f}")
    print(f"Macro F1        : {m.macro_f1:.4f}")
    print(f"Label Accuracy  : {m.label_accuracy:.4f}")
    print(f"Subset Accuracy : {m.subset_accuracy:.4f}")

    print("\nPer-aspect F1:")
    for a in target_aspects:
        print(f" - {a:<12} F1={m.per_aspect_f1[a]:.4f}  support_pos={m.support_pos[a]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--data-dir", default="data/raw/uit_visd4sa/data", help="Folder containing train/dev/test.jsonl")
    ap.add_argument("--dataset-yaml", default="config/datasets/dataset_a.yaml", help="Aspect mapping config")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick runs")

    # Baseline
    ap.add_argument("--baseline-dir", default="runs/baseline", help="Directory for BaselinePredictor")

    # PhoBERT Single
    ap.add_argument("--phobert-ckpt", required=True, help="Path to PhoBERT single checkpoint (.pt)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for PhoBERTSinglePredictor")

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_path = data_dir / f"{args.split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find split file: {data_path}")

    dataset_yaml = Path(args.dataset_yaml) if args.dataset_yaml else None
    target_aspects, aspect_mapping, ignored_aspects = load_dataset_config(dataset_yaml)

    # Import predictors from your repo
    from aspectmind.inference.baseline_predictor import BaselinePredictor
    from aspectmind.inference.phobert_single_predictor import PhoBERTSinglePredictor

    print(f"[INFO] Evaluating split: {args.split} at {data_path}")
    print(f"[INFO] target_aspects: {target_aspects}")
    print(f"[INFO] ignored_aspects: {sorted(list(ignored_aspects))}")

    # Baseline
    baseline = BaselinePredictor(args.baseline_dir)
    m_base = evaluate_predictor(
        baseline, data_path, target_aspects, aspect_mapping, ignored_aspects, args.max_samples
    )
    print_report("BASELINE RESULTS", m_base, target_aspects)

    # PhoBERT Single
    phobert = PhoBERTSinglePredictor(
        ckpt_path=args.phobert_ckpt,
        threshold=args.threshold,
    )
    m_pho = evaluate_predictor(
        phobert, data_path, target_aspects, aspect_mapping, ignored_aspects, args.max_samples
    )
    print_report("PHOBERT SINGLE RESULTS", m_pho, target_aspects)

    # Print a ready-to-copy Table 2 (use Micro F1, Label Accuracy, Micro Recall)
    print("\n" + "-" * 80)
    print("TABLE 2 (copy into report) — Preliminary Comparison Between Baseline and PhoBERT")
    print("-" * 80)
    print("| Model     | F1 (Micro) | Accuracy (Label) | Recall (Micro) |")
    print("|----------|------------:|-----------------:|---------------:|")
    print(f"| Baseline | {m_base.micro_f1:>10.4f} | {m_base.label_accuracy:>15.4f} | {m_base.micro_recall:>13.4f} |")
    print(f"| PhoBERT  | {m_pho.micro_f1:>10.4f} | {m_pho.label_accuracy:>15.4f} | {m_pho.micro_recall:>13.4f} |")


if __name__ == "__main__":
    main()