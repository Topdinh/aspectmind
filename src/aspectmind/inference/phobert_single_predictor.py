from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from transformers import AutoTokenizer

from aspectmind.models.phobert_single import PhoBERTSingleTask, ASPECTS


def _load_thresholds_json(path: Union[str, Path]) -> Dict[str, float]:
    """
    Load per-aspect thresholds from thresholds_*.json produced by scripts/tune_threshold_phobert_single.py.

    Expected structure (your current tuner output):
      {
        ...
        "best_per_aspect": {
          "thr": {"battery": 0.18, ...},
          ...
        },
        ...
      }

    Also supports a simple dict {aspect: thr, ...}.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    per = None
    if isinstance(data, dict):
        # primary expected format
        if "best_per_aspect" in data and isinstance(data["best_per_aspect"], dict):
            per = data["best_per_aspect"].get("thr", None)

        # fallback: file itself is the mapping
        if per is None:
            # accept a flat mapping if values are numeric
            if all(isinstance(v, (int, float)) for v in data.values()):
                per = data

    if not isinstance(per, dict):
        raise ValueError(f"Invalid thresholds file format: {p}")

    # enforce float
    return {str(k): float(v) for k, v in per.items()}


@dataclass
class PhoBERTSinglePredictor:
    """
    Load a trained PhoBERT single-task checkpoint and run inference for aspect detection (6 aspects).

    - ckpt_path: path to best_model.pt
    - threshold: default fallback threshold (0.5) for multi-label
    - thresholds_path: optional path to thresholds_*.json to enable per-aspect thresholds (calibration)
    - device: "cuda" or "cpu" or None(auto)

    Option B (Switchable Mode):
      - predict(text, use_calibrated=True/False)
        + True  -> use per-aspect thresholds if thresholds_path provided & loaded
        + False -> use global threshold (self.threshold)
    """

    ckpt_path: Union[str, Path] = Path("runs/phobert_single_2026-02-15_16-15-19/best_model.pt")
    threshold: float = 0.5
    thresholds_path: Optional[Union[str, Path]] = None
    device: Optional[str] = None  # "cuda" or "cpu" or None(auto)

    def __post_init__(self):
        self.ckpt_path = Path(self.ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.ckpt_path}")

        if self.device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device)

        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        # keep old behavior, but be a bit more robust:
        # - config may be missing
        # - model_state_dict might be nested in other keys (rare)
        self.cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        self.model_name = self.cfg.get("model_name", "vinai/phobert-base")
        self.max_length = int(self.cfg.get("max_length", 256))
        self.use_fast = bool(self.cfg.get("use_fast", False))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast)

        self.model = PhoBERTSingleTask(model_name=self.model_name, dropout=0.1)

        # Prefer your known key; fallback to other common formats without breaking old behavior
        state_dict = None
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            elif all(isinstance(k, str) for k in ckpt.keys()):
                # raw state_dict (unlikely for your runs but harmless)
                state_dict = ckpt

        if state_dict is None:
            raise KeyError("Checkpoint does not contain 'model_state_dict' (or compatible)")

        # ✅ IMPORTANT: strict=False for backward compatibility
        # Reason: new buffer 'temperature' was added later, old checkpoints won't have it.
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[phobert_single_predictor] Missing keys: {missing}")
        if unexpected:
            print(f"[phobert_single_predictor] Unexpected keys: {unexpected}")

        self.model.to(self._device)
        self.model.eval()

        # Load calibrated per-aspect thresholds if provided
        self.per_aspect_thresholds: Optional[Dict[str, float]] = None
        if self.thresholds_path is not None:
            tp = Path(self.thresholds_path)
            if not tp.exists():
                raise FileNotFoundError(f"Missing thresholds file: {tp}")
            self.per_aspect_thresholds = _load_thresholds_json(tp)

    @torch.no_grad()
    def predict_proba(self, text: str) -> Dict[str, float]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        out = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids", None),
            labels=None,
        )

        probs = torch.sigmoid(out["logits"])[0].detach().cpu().tolist()  # (6,)
        return {a: float(p) for a, p in zip(ASPECTS, probs)}

    def _get_threshold_for_aspect(self, aspect: str, use_calibrated: bool) -> float:
        """
        Return the decision threshold for a given aspect based on mode.
        """
        if use_calibrated and self.per_aspect_thresholds is not None:
            # fallback to global threshold if aspect missing in file
            return float(self.per_aspect_thresholds.get(aspect, self.threshold))
        return float(self.threshold)

    @torch.no_grad()
    def predict(self, text: str, use_calibrated: bool = True) -> Dict[str, int]:
        """
        Predict aspect presence (0/1) with switchable thresholding.

        - use_calibrated=True:
            use per-aspect thresholds from thresholds_path (if loaded)
        - use_calibrated=False:
            use global self.threshold (default 0.5)
        """
        proba = self.predict_proba(text)
        return {a: int(proba[a] >= self._get_threshold_for_aspect(a, use_calibrated)) for a in ASPECTS}