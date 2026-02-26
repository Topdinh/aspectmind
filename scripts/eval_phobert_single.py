# scripts/eval_phobert_single.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score

from aspectmind.data.loader import load_all_splits
from aspectmind.inference.phobert_single_predictor import PhoBERTSinglePredictor

ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


def labels_dict_to_vec(d: Dict[str, str]) -> np.ndarray:
    return np.array([1 if d.get(a, "not_mentioned") != "not_mentioned" else 0 for a in ASPECTS], dtype=np.int64)


def _resolve_path_optional(p: Optional[str], name: str) -> Optional[str]:
    if p is None:
        return None
    p = p.strip()
    if not p:
        return None
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return str(path)


def _load_temperature(temperature_path: str) -> float:
    """
    Expect JSON like:
      {"temperature": 0.8675, ...}
    """
    data = json.loads(Path(temperature_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "temperature" not in data:
        raise ValueError(f"Invalid temperature file: {temperature_path} (missing 'temperature')")
    T = float(data["temperature"])
    if not (T > 0.0):
        raise ValueError(f"Temperature must be > 0, got {T}")
    return T


def _bce_nll(probs: np.ndarray, y: np.ndarray) -> float:
    """
    probs,y: (N,A) in [0,1], {0,1}
    """
    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return float(nll)


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((probs - y) ** 2))


def _ece_binary_flat(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE for binary probs. For multi-label, flatten (N*A,)
    """
    p = probs.reshape(-1)
    t = y.reshape(-1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)

        if not np.any(mask):
            continue

        conf = float(np.mean(p[mask]))
        acc = float(np.mean(t[mask]))  # mean label in bin
        ece += abs(acc - conf) * (np.sum(mask) / n)

    return float(ece)


def _sigmoid_scaled(probs: Dict[str, float], T: float) -> Dict[str, float]:
    """
    Convert probabilities p -> logits -> logits/T -> sigmoid
    This is the correct temperature scaling transform for probabilities.
    """
    out: Dict[str, float] = {}
    eps = 1e-7
    for a, p in probs.items():
        p = float(np.clip(p, eps, 1.0 - eps))
        logit = np.log(p / (1.0 - p))
        logit_scaled = logit / max(float(T), 1e-6)
        p_scaled = 1.0 / (1.0 + np.exp(-logit_scaled))
        out[a] = float(p_scaled)
    return out


def _eval_once(
    data: List[dict],
    predictor: PhoBERTSinglePredictor,
    use_calibrated_thresholds: bool,
    temperature: Optional[float],
    ece_bins: int,
) -> Tuple[float, float, Dict[str, float], float, float, float]:
    """
    Returns:
      macro_f1, micro_f1, per_aspect_f1, nll, brier, ece
    """
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    y_prob_all: List[np.ndarray] = []

    per_aspect_true = {a: [] for a in ASPECTS}
    per_aspect_pred = {a: [] for a in ASPECTS}

    for ex in data:
        text = ex["text"]
        labels = ex["labels"]

        yt = labels_dict_to_vec(labels)

        # probs (optionally temperature-scaled for calibration metrics)
        proba = predictor.predict_proba(text)
        if temperature is not None:
            proba = _sigmoid_scaled(proba, T=temperature)

        # prediction (thresholding)
        if temperature is None:
            # keep old behavior: use predictor thresholds (global or per-aspect)
            pred = predictor.predict(text, use_calibrated=use_calibrated_thresholds)
        else:
            # temperature affects probabilities; for fair comparison, we threshold on scaled probs
            # but we must respect calibrated per-aspect thresholds if enabled.
            if use_calibrated_thresholds and getattr(predictor, "per_aspect_thresholds", None):
                thr_map = predictor.per_aspect_thresholds  # type: ignore[attr-defined]
                pred = {a: int(proba[a] >= float(thr_map.get(a, predictor.threshold))) for a in ASPECTS}
            else:
                pred = {a: int(proba[a] >= predictor.threshold) for a in ASPECTS}

        yp = np.array([pred[a] for a in ASPECTS], dtype=np.int64)
        yprob = np.array([proba[a] for a in ASPECTS], dtype=np.float64)

        y_true_all.append(yt)
        y_pred_all.append(yp)
        y_prob_all.append(yprob)

        for i, a in enumerate(ASPECTS):
            per_aspect_true[a].append(int(yt[i]))
            per_aspect_pred[a].append(int(yp[i]))

    y_true = np.stack(y_true_all, axis=0)  # (N,A)
    y_pred = np.stack(y_pred_all, axis=0)  # (N,A)
    y_prob = np.stack(y_prob_all, axis=0)  # (N,A)

    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_aspect_f1: Dict[str, float] = {}
    for a in ASPECTS:
        f1 = f1_score(per_aspect_true[a], per_aspect_pred[a], average="binary", zero_division=0)
        per_aspect_f1[a] = float(f1)

    # calibration metrics computed on probabilities
    nll = _bce_nll(y_prob, y_true.astype(np.float64))
    brier = _brier(y_prob, y_true.astype(np.float64))
    ece = _ece_binary_flat(y_prob, y_true.astype(np.float64), n_bins=ece_bins)

    return float(macro), float(micro), per_aspect_f1, float(nll), float(brier), float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, required=True, help="Path tới best_model.pt của PhoBERT single")
    ap.add_argument("--threshold", type=float, default=0.5, help="Global threshold (used if thresholds_path not set)")
    ap.add_argument("--thresholds_path", type=str, default="", help="Path tới thresholds_*.json (per-aspect calibrated)")
    ap.add_argument("--temperature_path", type=str, default="", help="Path tới temperature_*.json (temperature scaling)")
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    args = ap.parse_args()

    thresholds_path = _resolve_path_optional(args.thresholds_path, "thresholds_path")
    temperature_path = _resolve_path_optional(args.temperature_path, "temperature_path")

    data = load_all_splits()[args.split]

    predictor = PhoBERTSinglePredictor(
        ckpt_path=args.ckpt_path,
        threshold=args.threshold,
        thresholds_path=thresholds_path,
    )

    # threshold mode
    use_calibrated_thresholds = thresholds_path is not None
    thr_info = (
        f"calibrated(per-aspect) from {thresholds_path}"
        if use_calibrated_thresholds
        else f"global={args.threshold}"
    )

    print(f"\n[EVAL:{args.split.upper()}] PHOBERT_SINGLE ckpt={args.ckpt_path}")
    print(f"  threshold_mode : {thr_info}")

    # ---- BEFORE (T=1) ----
    macro, micro, per_f1, nll, brier, ece = _eval_once(
        data=data,
        predictor=predictor,
        use_calibrated_thresholds=use_calibrated_thresholds,
        temperature=None,
        ece_bins=args.ece_bins,
    )

    print("\n--- BEFORE Temperature Scaling (T=1.0) ---")
    print(f"  macro_f1 : {macro:.4f}")
    print(f"  micro_f1 : {micro:.4f}")
    print(f"  nll_bce  : {nll:.6f}")
    print(f"  brier    : {brier:.6f}")
    print(f"  ece({args.ece_bins} bins): {ece:.6f}")
    print("  per-aspect F1:")
    for a in ASPECTS:
        print(f"    - {a:11s}: {per_f1.get(a, 0.0):.4f}")

    # ---- AFTER (T from file) ----
    if temperature_path is not None:
        T = _load_temperature(temperature_path)

        macro2, micro2, per_f12, nll2, brier2, ece2 = _eval_once(
            data=data,
            predictor=predictor,
            use_calibrated_thresholds=use_calibrated_thresholds,
            temperature=T,
            ece_bins=args.ece_bins,
        )

        print("\n--- AFTER Temperature Scaling ---")
        print(f"  temperature : {T:.6f}  (from {temperature_path})")
        print(f"  macro_f1 : {macro2:.4f}")
        print(f"  micro_f1 : {micro2:.4f}")
        print(f"  nll_bce  : {nll2:.6f}")
        print(f"  brier    : {brier2:.6f}")
        print(f"  ece({args.ece_bins} bins): {ece2:.6f}")
        print("  per-aspect F1:")
        for a in ASPECTS:
            print(f"    - {a:11s}: {per_f12.get(a, 0.0):.4f}")

        # deltas
        print("\n--- DELTA (AFTER - BEFORE) ---")
        print(f"  Δmacro_f1 : {macro2 - macro:+.4f}")
        print(f"  Δmicro_f1 : {micro2 - micro:+.4f}")
        print(f"  Δnll_bce  : {nll2 - nll:+.6f}")
        print(f"  Δbrier    : {brier2 - brier:+.6f}")
        print(f"  Δece      : {ece2 - ece:+.6f}")

    else:
        print("\n[INFO] temperature_path not provided -> only BEFORE metrics printed.")


if __name__ == "__main__":
    main()