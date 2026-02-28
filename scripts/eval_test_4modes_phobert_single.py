from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch, ASPECTS
from aspectmind.models.phobert_single import PhoBERTSingleTask


# ===============================
# Calibration metrics
# ===============================

def nll_bce(probs: np.ndarray, targets: np.ndarray) -> float:
    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs)))


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def ece_binary(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.flatten()
    labels = targets.flatten()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / total) * abs(acc - conf)

    return float(ece)


# ===============================
# Utils
# ===============================

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_first_key(cfg: Dict[str, Any], keys: list[str], default: Any) -> Any:
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def _get_attr_any(obj: Any, names: list[str]) -> Optional[Any]:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _infer_num_labels_from_dataset(dataset) -> int:
    sample = dataset[0]
    if isinstance(sample, dict) and "labels" in sample:
        return int(len(sample["labels"]))
    raise RuntimeError("Cannot infer num_labels: dataset[0] does not contain key 'labels'.")


def _extract_logits(model_out: Any) -> torch.Tensor:
    if isinstance(model_out, torch.Tensor):
        return model_out

    if isinstance(model_out, dict):
        for k in ("logits", "logit"):
            if k in model_out and isinstance(model_out[k], torch.Tensor):
                return model_out[k]
        for v in model_out.values():
            if isinstance(v, torch.Tensor):
                return v
        raise TypeError(f"Model returned dict but no tensor found. Keys={list(model_out.keys())}")

    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        if isinstance(model_out[0], torch.Tensor):
            return model_out[0]
        raise TypeError(f"Model returned tuple/list but first element is not Tensor: {type(model_out[0])}")

    raise TypeError(f"Unsupported model output type: {type(model_out)}")


def _parse_thresholds(thr_json: Dict[str, Any], num_labels: int) -> np.ndarray:
    """
    Support multiple formats for thresholds_dev.json:

    Common:
      thr_json["best_per_aspect"]["thr"] = [t1..t6]

    Also possible:
      thr_json["best_per_aspect"]["thr"] = {"battery":0.4, "camera":0.5, ...}

    We map dict thresholds using ASPECTS order (stable in this project).
    """
    if "best_per_aspect" not in thr_json:
        raise KeyError("thresholds_dev.json missing key: best_per_aspect")
    best = thr_json["best_per_aspect"]
    if "thr" not in best:
        raise KeyError("thresholds_dev.json missing key: best_per_aspect.thr")

    thr = best["thr"]

    # Case 1: list/tuple
    if isinstance(thr, (list, tuple, np.ndarray)):
        arr = np.array(thr, dtype=np.float32)
        if arr.shape[0] != num_labels:
            raise ValueError(f"threshold list length mismatch: got {arr.shape[0]} but num_labels={num_labels}")
        return arr

    # Case 2: dict mapping aspect -> threshold
    if isinstance(thr, dict):
        # expect keys include aspect names (battery/camera/...)
        missing = [a for a in ASPECTS[:num_labels] if a not in thr]
        if missing:
            raise ValueError(f"threshold dict missing aspects: {missing}. Keys={list(thr.keys())}")

        arr = np.array([float(thr[a]) for a in ASPECTS[:num_labels]], dtype=np.float32)
        return arr

    raise TypeError(f"Unsupported thresholds format at best_per_aspect.thr: {type(thr)}")


def _build_test_bundle_tokenizer_numlabels(run_dir: Path) -> Tuple[Any, Any, int, Dict[str, Any]]:
    config_path = run_dir / "config.json"
    cfg: Dict[str, Any] = _load_json(config_path) if config_path.exists() else {}

    model_name = _get_first_key(
        cfg,
        keys=["model_name", "hf_model", "backbone", "pretrained_model", "transformer_model"],
        default="vinai/phobert-base",
    )
    max_length = int(_get_first_key(cfg, keys=["max_length", "max_len", "max_seq_len"], default=256))
    use_fast = bool(_get_first_key(cfg, keys=["use_fast", "tokenizer_use_fast"], default=False))

    bundle = build_phobert_datasets(model_name=model_name, max_length=max_length, use_fast=use_fast)

    tokenizer = _get_attr_any(bundle, ["tokenizer"])
    if tokenizer is None:
        raise RuntimeError("Could not find tokenizer in DataBundle (expected attribute tokenizer).")

    test_dataset = _get_attr_any(bundle, ["test_dataset", "test", "test_ds"])
    if test_dataset is None:
        raise RuntimeError("Could not find test dataset in DataBundle (expected attribute test_dataset/test).")

    aspects = _get_attr_any(bundle, ["target_aspects", "aspects", "label_names", "labels"])
    if aspects is not None and hasattr(aspects, "__len__"):
        num_labels = len(aspects)
    else:
        num_labels = _infer_num_labels_from_dataset(test_dataset)

    meta = {"model_name": model_name, "max_length": max_length, "use_fast": use_fast}
    return bundle, tokenizer, num_labels, meta


def _init_model(num_labels: int, meta: Dict[str, Any]) -> torch.nn.Module:
    sig = inspect.signature(PhoBERTSingleTask)
    kwargs: Dict[str, Any] = {}

    if "num_labels" in sig.parameters:
        kwargs["num_labels"] = num_labels
    if "model_name" in sig.parameters:
        kwargs["model_name"] = meta.get("model_name", "vinai/phobert-base")
    if "max_length" in sig.parameters:
        kwargs["max_length"] = meta.get("max_length", 256)
    if "use_fast" in sig.parameters:
        kwargs["use_fast"] = meta.get("use_fast", False)

    return PhoBERTSingleTask(**kwargs)


def collect_logits(
    model: torch.nn.Module,
    dataset,
    tokenizer,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_batch(tokenizer, b),
    )

    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = _extract_logits(out)

            all_logits.append(logits.detach().cpu())
            all_targets.append(labels.detach().cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    targets_np = torch.cat(all_targets, dim=0).numpy()
    return logits_np, targets_np


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probs >= thresholds).astype(int)


# ===============================
# Main
# ===============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    bundle, tokenizer, num_labels, meta = _build_test_bundle_tokenizer_numlabels(run_dir)
    test_dataset = _get_attr_any(bundle, ["test_dataset", "test", "test_ds"])

    model = _init_model(num_labels=num_labels, meta=meta)

    ckpt_path = run_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    missing_real = [k for k in missing if k != "temperature"]
    if missing_real:
        raise RuntimeError(f"Unexpected missing keys in checkpoint: {missing_real}")
    if unexpected:
        print(f"[INFO] Unexpected keys in checkpoint (ignored): {unexpected}")

    model.to(device)

    logits, targets = collect_logits(model, test_dataset, tokenizer, device)

    thr_json = _load_json(run_dir / "thresholds_dev.json")
    temp_json = _load_json(run_dir / "temperature_dev.json")

    thr_global = 0.5
    thr_tuned = _parse_thresholds(thr_json, num_labels=num_labels)
    T = float(temp_json["temperature"])

    results: Dict[str, Dict[str, float]] = {}

    modes = {
        "A_global": {"thr": np.full(num_labels, thr_global, dtype=np.float32), "temp": False},
        "B_tuned": {"thr": thr_tuned, "temp": False},
        "C_global_temp": {"thr": np.full(num_labels, thr_global, dtype=np.float32), "temp": True},
        "D_tuned_temp": {"thr": thr_tuned, "temp": True},
    }

    logits_t = torch.from_numpy(logits).float()

    for name, setting in modes.items():
        logits_used = logits_t.clone()
        if setting["temp"]:
            logits_used = logits_used / T

        probs = torch.sigmoid(logits_used).cpu().numpy()
        preds = apply_thresholds(probs, setting["thr"])

        macro = f1_score(targets, preds, average="macro", zero_division=0)
        micro = f1_score(targets, preds, average="micro", zero_division=0)

        results[name] = {
            "macro_f1": float(macro),
            "micro_f1": float(micro),
            "nll": nll_bce(probs, targets),
            "brier": brier_score(probs, targets),
            "ece": ece_binary(probs, targets),
        }

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()