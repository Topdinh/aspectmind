from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch
from aspectmind.eval.threshold_tuning import ASPECTS
from aspectmind.models.phobert_single import PhoBERTSingleTask


# ----------------------------
# Robust checkpoint loader
# ----------------------------
def load_single_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt)}")

    state = None
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif all(isinstance(k, str) for k in ckpt.keys()):
        # raw state_dict (rare)
        state = ckpt

    if state is None:
        raise KeyError("Checkpoint does not contain model_state_dict/state_dict")

    # strip "model." prefix if exists
    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model.") :]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[temp_scale] Missing keys: {missing}")
    if unexpected:
        print(f"[temp_scale] Unexpected keys: {unexpected}")

    return ckpt


# ----------------------------
# Patch load_split signature drift (safe)
# ----------------------------
def _default_sentiment_mapping() -> Dict[str, int]:
    return {
        "pos": 0,
        "positive": 0,
        "POS": 0,
        "Pos": 0,
        "neg": 1,
        "negative": 1,
        "NEG": 1,
        "Neg": 1,
        "neu": 2,
        "neutral": 2,
        "NEU": 2,
        "Neu": 2,
    }


def _patch_load_split_everywhere(target_aspects: List[str]) -> None:
    """
    Patch load_split references so build_phobert_datasets() won't crash if signature drift exists.
    We patch:
      - aspectmind.data.loader.load_split
      - aspectmind.data.transformer_dataset.load_split (if imported directly)
    """
    candidates = [
        "aspectmind.data.loader",
        "aspectmind.data.transformer_dataset",
    ]

    sentiment_mapping = _default_sentiment_mapping()
    ignored_aspects: List[str] = []

    patched_any = False

    for mod_name in candidates:
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue

        if not hasattr(m, "load_split") or not callable(getattr(m, "load_split")):
            continue

        original = getattr(m, "load_split")
        if getattr(original, "_aspectmind_patched", False):
            continue

        sig = inspect.signature(original)

        def wrapper(*args, __orig=original, __sig=sig, **kwargs):
            """
            Safe wrapper:
            - try direct call first
            - if TypeError, bind partial and fill missing known params by name
            - call original with kwargs-only to avoid "multiple values" collisions
            """
            try:
                return __orig(*args, **kwargs)
            except TypeError:
                pass

            # bind partial
            bound = __sig.bind_partial(*args, **kwargs)
            call_kwargs = dict(bound.arguments)

            # fill missing
            for name, p in __sig.parameters.items():
                if name in call_kwargs:
                    continue
                lname = name.lower()

                if ("target" in lname and "aspect" in lname) or lname == "target_aspects":
                    call_kwargs[name] = target_aspects
                elif "sentiment" in lname:
                    call_kwargs[name] = sentiment_mapping
                elif "ignored" in lname:
                    call_kwargs[name] = ignored_aspects
                else:
                    # if required but unknown -> try default if exists, else None
                    if p.default is not inspect._empty:
                        call_kwargs[name] = p.default
                    else:
                        call_kwargs[name] = None

            # kwargs-only to avoid duplicate positional/keyword for things like file_path
            return __orig(**call_kwargs)

        setattr(wrapper, "_aspectmind_patched", True)
        setattr(m, "load_split", wrapper)
        patched_any = True

    if patched_any:
        print("[temp_scale] Patched load_split in: aspectmind.data.loader / transformer_dataset (if needed)")


# ----------------------------
# Metrics: NLL (BCE), Brier, ECE
# ----------------------------
def _bce_nll(probs: torch.Tensor, y: torch.Tensor) -> float:
    """
    NLL for multi-label = BCE on probabilities.
    probs,y: (N,A)
    """
    eps = 1e-7
    probs = torch.clamp(probs, eps, 1 - eps)
    nll = -(y * torch.log(probs) + (1 - y) * torch.log(1 - probs)).mean()
    return float(nll.detach().cpu().item())


def _brier(probs: torch.Tensor, y: torch.Tensor) -> float:
    return float(((probs - y) ** 2).mean().detach().cpu().item())


def _ece_binary_flat(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """
    ECE for binary probabilities.
    For multi-label, we flatten (N*A,) and treat each as a binary sample.
    """
    p = probs.reshape(-1).detach().cpu().numpy()
    t = y.reshape(-1).detach().cpu().numpy()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(t[mask]))  # for binary label, mean label in bin
        ece += abs(acc - conf) * (np.sum(mask) / n)
    return float(ece)


def _f1_from_probs(probs: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> Tuple[float, float]:
    """
    Return (macro_f1, micro_f1) with global threshold thr.
    probs,y: (N,A)
    """
    yp = (probs >= thr).long().detach().cpu().numpy()
    yt = y.long().detach().cpu().numpy()
    micro = f1_score(yt.reshape(-1), yp.reshape(-1), average="micro", zero_division=0)
    macro = f1_score(yt, yp, average="macro", zero_division=0)
    return float(macro), float(micro)


# ----------------------------
# Collect logits/labels
# ----------------------------
@torch.no_grad()
def collect_logits_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_aspects: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits: (N, A) float
      y:      (N, A) float in {0,1}
    """
    model.eval()
    all_logits = []
    all_y = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
        else:
            raise TypeError("Model output must be dict with key 'logits'")

        if logits.dim() != 2 or logits.size(-1) != num_aspects:
            raise ValueError(f"Unexpected logits shape {tuple(logits.shape)} expected (N,{num_aspects})")

        y = batch["labels"].to(device).float()
        # clamp ignore -1 to 0 for calibration (shouldn't exist, but safe)
        y = torch.clamp(y, 0.0, 1.0)

        all_logits.append(logits)
        all_y.append(y)

    logits_all = torch.cat(all_logits, dim=0)
    y_all = torch.cat(all_y, dim=0)
    return logits_all, y_all


# ----------------------------
# Fit Temperature T on DEV
# ----------------------------
def fit_temperature(
    logits: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    max_iter: int = 50,
) -> float:
    """
    Learn scalar temperature T > 0 minimizing BCEWithLogitsLoss(logits / T, y).
    Uses LBFGS on logT parameter for stability.
    """
    logits = logits.detach().to(device)
    y = y.detach().to(device)

    logT = torch.nn.Parameter(torch.zeros(1, device=device))
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS([logT], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(logT)  # >0
        scaled = logits / torch.clamp(T, min=1e-6)
        loss = loss_fn(scaled, y)
        loss.backward()
        return loss

    optimizer.step(closure)

    T_final = float(torch.exp(logT).detach().cpu().item())
    return T_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="runs/.../best_model.pt")
    ap.add_argument("--split", default="dev", choices=["dev"], help="Temperature is fit on DEV only")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="vinai/phobert-base", help="tokenizer/model base name")
    ap.add_argument("--out_dir", type=str, default="", help="where to save temperature_dev.json (default: same dir as ckpt)")
    ap.add_argument("--max_iter", type=int, default=50, help="LBFGS max_iter")
    ap.add_argument("--ece_bins", type=int, default=15)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Patch compat BEFORE building datasets (safe no-op if not needed)
    _patch_load_split_everywhere(target_aspects=list(ASPECTS))

    # Load ckpt config to use right tokenizer settings if present
    # (still fallback to args)
    tmp_model = PhoBERTSingleTask(model_name=args.ckpt, dropout=0.1)
    ckpt = load_single_checkpoint(tmp_model, ckpt_path)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_name = cfg.get("model_name", args.ckpt)
    max_length = int(cfg.get("max_length", args.max_length))
    use_fast = bool(cfg.get("use_fast", False))

    # Rebuild model with correct base
    model = PhoBERTSingleTask(model_name=model_name, dropout=0.1).to(device)
    load_single_checkpoint(model, ckpt_path)

    # Build dataset bundle
    bundle = build_phobert_datasets(
        model_name=model_name,
        max_length=max_length,
        use_fast=use_fast,
    )
    ds = bundle.dev_dataset

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
        num_workers=0,
    )

    # Collect logits/y
    logits, y = collect_logits_and_labels(model, loader, device, num_aspects=len(ASPECTS))

    # Metrics before
    probs_before = torch.sigmoid(logits)
    nll_before = _bce_nll(probs_before, y)
    brier_before = _brier(probs_before, y)
    ece_before = _ece_binary_flat(probs_before, y, n_bins=args.ece_bins)
    macro_before, micro_before = _f1_from_probs(probs_before, y, thr=0.5)

    # Fit T
    T = fit_temperature(logits, y, device=device, max_iter=args.max_iter)

    # Metrics after
    logits_after = logits / max(T, 1e-6)
    probs_after = torch.sigmoid(logits_after)
    nll_after = _bce_nll(probs_after, y)
    brier_after = _brier(probs_after, y)
    ece_after = _ece_binary_flat(probs_after, y, n_bins=args.ece_bins)
    macro_after, micro_after = _f1_from_probs(probs_after, y, thr=0.5)

    print("\n=== TEMPERATURE SCALING (FIT ON DEV) ===")
    print(f"T = {T:.6f}")

    print("\n--- DEV metrics (Before -> After) ---")
    print(f"NLL(BCE)   : {nll_before:.6f} -> {nll_after:.6f}")
    print(f"Brier      : {brier_before:.6f} -> {brier_after:.6f}")
    print(f"ECE({args.ece_bins} bins): {ece_before:.6f} -> {ece_after:.6f}")
    print(f"macro_f1@0.5 : {macro_before:.4f} -> {macro_after:.4f}")
    print(f"micro_f1@0.5 : {micro_before:.4f} -> {micro_after:.4f}")

    # Save
    out_dir = Path(args.out_dir) if args.out_dir.strip() else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "temperature_dev.json"

    payload: Dict[str, object] = {
        "ckpt_path": str(ckpt_path),
        "model_name": model_name,
        "max_length": max_length,
        "temperature": T,
        "fit_split": "dev",
        "metrics_dev_before": {
            "nll_bce": nll_before,
            "brier": brier_before,
            "ece": ece_before,
            "macro_f1@0.5": macro_before,
            "micro_f1@0.5": micro_before,
        },
        "metrics_dev_after": {
            "nll_bce": nll_after,
            "brier": brier_after,
            "ece": ece_after,
            "macro_f1@0.5": macro_after,
            "micro_f1@0.5": micro_after,
        },
        "aspects": ASPECTS,
        "ece_bins": args.ece_bins,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_json}")


if __name__ == "__main__":
    main()