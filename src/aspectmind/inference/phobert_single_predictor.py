from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoTokenizer

# Reduce noisy HF/Transformers logs (best-effort, harmless if versions differ)
try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass

from aspectmind.models.phobert_single import PhoBERTSingleTask, ASPECTS


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_logits(model_out: Any) -> torch.Tensor:
    """
    PhoBERTSingleTask output in this project may be:
      - torch.Tensor (logits)
      - dict with key 'logits'
      - tuple/list where first element is logits
    """
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, dict):
        if "logits" in model_out and isinstance(model_out["logits"], torch.Tensor):
            return model_out["logits"]
        # fallback: first tensor value in dict
        for v in model_out.values():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError(f"Model returned dict but no tensor found. Keys={list(model_out.keys())}")

    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]
        raise TypeError(f"Model returned tuple/list but first element is not Tensor: {type(model_out[0])}")

    raise TypeError(f"Unsupported model output type: {type(model_out)}")


def _parse_thresholds(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Supports thresholds JSON produced by your tuner.

    Supported formats:
      1) { "best_per_aspect": { "thr": {"battery":0.18, ...} } }
      2) { "best_per_aspect": { "thr": [0.18, 0.22, ...] } }
      3) Flat dict: {"battery":0.18, ...}
    """
    per = None

    if isinstance(data, dict):
        if "best_per_aspect" in data and isinstance(data["best_per_aspect"], dict):
            per = data["best_per_aspect"].get("thr", None)

        if per is None:
            # flat mapping
            if all(isinstance(v, (int, float)) for v in data.values()):
                per = data

    # per can be dict or list
    if isinstance(per, dict):
        return {str(k): float(v) for k, v in per.items()}

    if isinstance(per, (list, tuple)):
        if len(per) != len(ASPECTS):
            raise ValueError(f"Threshold list length mismatch: got {len(per)} but expected {len(ASPECTS)}")
        return {a: float(t) for a, t in zip(ASPECTS, per)}

    raise ValueError("Invalid thresholds file format (expected best_per_aspect.thr as dict/list or a flat dict).")


def _load_temperature(data: Dict[str, Any]) -> float:
    """
    temperature_dev.json expected:
      {"temperature": 0.8675, ...}
    """
    if not isinstance(data, dict) or "temperature" not in data:
        raise ValueError("Invalid temperature file format: missing key 'temperature'")
    return float(data["temperature"])


def _init_model_robust(model_name: str, num_labels: int = 6) -> PhoBERTSingleTask:
    """
    PhoBERTSingleTask signature may vary across edits.
    We pass only supported kwargs to avoid breaking.
    """
    sig = inspect.signature(PhoBERTSingleTask)
    kwargs: Dict[str, Any] = {}

    if "model_name" in sig.parameters:
        kwargs["model_name"] = model_name
    if "num_labels" in sig.parameters:
        kwargs["num_labels"] = num_labels
    # keep your old default but only if supported
    if "dropout" in sig.parameters:
        kwargs["dropout"] = 0.1

    return PhoBERTSingleTask(**kwargs)


@dataclass
class PhoBERTSinglePredictor:
    """
    Load a trained PhoBERT single-task checkpoint and run inference for aspect detection (6 aspects).

    - ckpt_path: path to best_model.pt
    - threshold: global threshold fallback (default 0.5)
    - thresholds_path: optional path to thresholds_*.json (per-aspect thresholds)
    - temperature_path: optional path to temperature_*.json (temperature scaling)
    - device: "cuda" or "cpu" or None(auto)

    Switchable modes:
      - predict(text, use_calibrated=True/False, use_temperature=True/False)
      - predict_proba(text, use_temperature=True/False)

    Notes:
      - If thresholds_path is None, auto-detect run_dir/thresholds_dev.json.
      - If temperature_path is None, auto-detect run_dir/temperature_dev.json.
      - If calibration files missing, falls back safely to global threshold and no temperature scaling.
    """

    ckpt_path: Union[str, Path] = Path("runs/phobert_single_2026-02-15_16-15-19/best_model.pt")
    threshold: float = 0.5
    thresholds_path: Optional[Union[str, Path]] = None
    temperature_path: Optional[Union[str, Path]] = None
    device: Optional[str] = None  # "cuda" or "cpu" or None(auto)

    def __post_init__(self):
        self.ckpt_path = Path(self.ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.ckpt_path}")

        # resolve device
        if self.device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device)

        # run_dir is folder containing best_model.pt
        self.run_dir = self.ckpt_path.parent

        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        # read config (best effort)
        self.cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        self.model_name = self.cfg.get("model_name", "vinai/phobert-base")
        self.max_length = int(self.cfg.get("max_length", 256))
        self.use_fast = bool(self.cfg.get("use_fast", False))

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=self.use_fast)

        # model
        self.model = _init_model_robust(model_name=self.model_name, num_labels=len(ASPECTS))

        # state_dict extraction (robust)
        state_dict = None
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            elif all(isinstance(k, str) for k in ckpt.keys()):
                state_dict = ckpt  # raw state_dict fallback

        if state_dict is None:
            raise KeyError("Checkpoint does not contain 'model_state_dict' (or compatible)")

        # ✅ load with strict=False (temperature buffer might not exist in old ckpt)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        # Only warn for REAL missing keys (ignore 'temperature' quietly)
        missing_real = [k for k in missing if k != "temperature"]
        if missing_real:
            # keep it short + meaningful (no spam)
            print(f"[PhoBERTSinglePredictor] Warning: missing keys in checkpoint: {missing_real}")
        if unexpected:
            # unexpected often happens if ckpt contains extra; keep short
            print(f"[PhoBERTSinglePredictor] Info: unexpected keys ignored: {unexpected}")

        self.model.to(self._device)
        self.model.eval()

        # ---- Load calibrated thresholds (optional, safe fallback)
        self.per_aspect_thresholds: Optional[Dict[str, float]] = None
        tp = Path(self.thresholds_path) if self.thresholds_path is not None else (self.run_dir / "thresholds_dev.json")
        if tp.exists():
            try:
                thr_data = _read_json(tp)
                self.per_aspect_thresholds = _parse_thresholds(thr_data)
            except Exception as e:
                # safe fallback
                print(f"[PhoBERTSinglePredictor] Warning: failed to load thresholds from {tp}: {e}")
                self.per_aspect_thresholds = None

        # ---- Load temperature (optional, safe fallback)
        self.temperature: Optional[float] = None
        tpath = Path(self.temperature_path) if self.temperature_path is not None else (self.run_dir / "temperature_dev.json")
        if tpath.exists():
            try:
                t_data = _read_json(tpath)
                self.temperature = _load_temperature(t_data)
            except Exception as e:
                print(f"[PhoBERTSinglePredictor] Warning: failed to load temperature from {tpath}: {e}")
                self.temperature = None

    def _get_threshold_for_aspect(self, aspect: str, use_calibrated: bool) -> float:
        if use_calibrated and self.per_aspect_thresholds is not None:
            return float(self.per_aspect_thresholds.get(aspect, self.threshold))
        return float(self.threshold)

    @torch.no_grad()
    def predict_proba(self, text: str, use_temperature: bool = False) -> Dict[str, float]:
        """
        Return probabilities per aspect.

        - use_temperature=False (default): standard sigmoid(logits)
        - use_temperature=True: apply logits / T before sigmoid if temperature is available
          (if temperature missing, falls back silently to raw logits)
        """
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

        logits = _extract_logits(out)  # shape [B, 6]
        if use_temperature and (self.temperature is not None) and (self.temperature > 0):
            logits = logits / float(self.temperature)

        probs = torch.sigmoid(logits)[0].detach().cpu().tolist()
        return {a: float(p) for a, p in zip(ASPECTS, probs)}

    @torch.no_grad()
    def predict(self, text: str, use_calibrated: bool = True, use_temperature: bool = False) -> Dict[str, int]:
        """
        Predict aspect presence (0/1).

        - use_calibrated=True: use per-aspect thresholds if available
        - use_calibrated=False: use global threshold self.threshold
        - use_temperature=True: apply temperature scaling on logits before sigmoid (if available)
        """
        proba = self.predict_proba(text, use_temperature=use_temperature)
        return {a: int(proba[a] >= self._get_threshold_for_aspect(a, use_calibrated)) for a in ASPECTS}

    def status(self) -> Dict[str, Any]:
        """
        Small helper for UI/debug: show what is loaded.
        """
        return {
            "device": str(self._device),
            "model_name": self.model_name,
            "max_length": self.max_length,
            "use_fast": self.use_fast,
            "global_threshold": float(self.threshold),
            "has_per_aspect_thresholds": self.per_aspect_thresholds is not None,
            "temperature": self.temperature,
            "run_dir": str(self.run_dir),
        }