from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score


ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


@dataclass
class TuneResult:
    mode: str  # "global" or "per_aspect"
    global_thr: float | None
    per_aspect_thr: Dict[str, float] | None
    macro_f1: float
    micro_f1: float
    per_aspect_f1: Dict[str, float]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _compute_scores(y_true: np.ndarray, y_prob: np.ndarray, thr: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """
    y_true: (N,6) {0,1}
    y_prob: (N,6) [0,1]
    thr:    (6,) thresholds
    """
    y_pred = (y_prob >= thr.reshape(1, -1)).astype(int)

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    per = {}
    for i, a in enumerate(ASPECTS):
        per[a] = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
    return float(macro), float(micro), per


def tune_thresholds_from_probs(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    grid: List[float] | None = None,
) -> Tuple[TuneResult, TuneResult]:
    """
    Returns:
      - best_global (one threshold for all aspects)
      - best_per_aspect (threshold per aspect)
    """
    if grid is None:
        grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]

    y_true_np = _to_numpy(y_true).astype(int)
    y_prob_np = _to_numpy(y_prob).astype(float)

    # ---- Global threshold ----
    best_g = None
    for t in grid:
        thr = np.array([t] * len(ASPECTS), dtype=float)
        macro, micro, per = _compute_scores(y_true_np, y_prob_np, thr)
        if (best_g is None) or (macro > best_g.macro_f1):
            best_g = TuneResult(
                mode="global",
                global_thr=float(t),
                per_aspect_thr=None,
                macro_f1=macro,
                micro_f1=micro,
                per_aspect_f1=per,
            )

    # ---- Per-aspect thresholds (independent maximize per-aspect F1) ----
    best_thr = {}
    for i, a in enumerate(ASPECTS):
        best_a = (0.5, -1.0)  # (thr, f1)
        for t in grid:
            pred = (y_prob_np[:, i] >= t).astype(int)
            f1 = f1_score(y_true_np[:, i], pred, average="binary", zero_division=0)
            if f1 > best_a[1]:
                best_a = (float(t), float(f1))
        best_thr[a] = best_a[0]

    thr_vec = np.array([best_thr[a] for a in ASPECTS], dtype=float)
    macro, micro, per = _compute_scores(y_true_np, y_prob_np, thr_vec)
    best_p = TuneResult(
        mode="per_aspect",
        global_thr=None,
        per_aspect_thr=best_thr,
        macro_f1=macro,
        micro_f1=micro,
        per_aspect_f1=per,
    )

    return best_g, best_p
