from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch

ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


# -------- checkpoint loader (robust like your tuning script) --------
def load_single_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model.") :]: v for k, v in state.items()}

    model.load_state_dict(state, strict=False)


def resolve_single_model_class():
    import importlib
    import torch.nn as nn

    mod = importlib.import_module("aspectmind.models.phobert_single")

    # try common names first
    common = [
        "PhoBERTSingle",
        "PhoBERTSingleModel",
        "PhoBERTSingleTask",
        "PhoBERTSingleClassifier",
        "PhoBERTAspectClassifier",
        "PhoBERTForAspects",
        "PhoBERTForAspect",
    ]
    for name in common:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls

    # fallback: pick first nn.Module class defined in module
    candidates = []
    for _, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj.__module__ == mod.__name__:
            candidates.append(obj)

    if not candidates:
        raise ImportError("No torch.nn.Module subclass found in aspectmind.models.phobert_single")

    candidates.sort(key=lambda c: (("PhoBERT" not in c.__name__), c.__name__))
    return candidates[0]


def instantiate_single_model(model_cls: type, ckpt: str, num_aspects: int) -> torch.nn.Module:
    tries = [
        dict(ckpt=ckpt, num_aspects=num_aspects),
        dict(model_name=ckpt, num_aspects=num_aspects),
        dict(ckpt=ckpt),
        dict(model_name=ckpt),
        {},
    ]
    last_err: Optional[Exception] = None
    for kwargs in tries:
        try:
            return model_cls(**kwargs)
        except TypeError as e:
            last_err = e
    raise TypeError(f"Cannot instantiate {model_cls.__name__}. Last error: {last_err}")


def extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    if isinstance(out, dict):
        for k in ["logits", "aspect_logits", "pred_logits"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
    raise TypeError(f"Unsupported output type: {type(out)}")


def ensure_logits_shape(logits: torch.Tensor, num_aspects: int) -> torch.Tensor:
    if logits.dim() == 2 and logits.size(-1) == num_aspects:
        return logits
    if logits.dim() == 3:
        if logits.size(1) == num_aspects and logits.size(2) == 1:
            return logits.squeeze(-1)
        if logits.size(2) == num_aspects and logits.size(1) == 1:
            return logits.squeeze(1)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}; expected (B,{num_aspects}).")


@torch.no_grad()
def collect_probs_and_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_aspects: int):
    all_true = []
    all_prob = []

    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = ensure_logits_shape(extract_logits(out), num_aspects)
        probs = torch.sigmoid(logits).detach().cpu()

        y = batch["labels"].detach().cpu().long()
        # ignore -> clamp to 0/1
        y = torch.clamp(y, 0, 1)

        all_true.append(y)
        all_prob.append(probs)

    y_true = torch.cat(all_true, dim=0)  # (N, A)
    y_prob = torch.cat(all_prob, dim=0)  # (N, A)
    return y_true.numpy(), y_prob.numpy()


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return float(2 * prec * rec / (prec + rec + 1e-12))


def compute_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    return f1_from_counts(tp, fp, fn)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s: List[float] = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = float(((yt == 1) & (yp == 1)).sum())
        fp = float(((yt == 0) & (yp == 1)).sum())
        fn = float(((yt == 1) & (yp == 0)).sum())
        f1s.append(f1_from_counts(tp, fp, fn))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="runs/.../best_model.pt")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="vinai/phobert-base")
    ap.add_argument("--out_dir", type=str, default="", help="default: same folder as ckpt")
    ap.add_argument("--thr_start", type=float, default=0.10)
    ap.add_argument("--thr_end", type=float, default=0.90)
    ap.add_argument("--thr_step", type=float, default=0.01)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build datasets
    bundle = build_phobert_datasets(model_name=args.ckpt, max_length=args.max_length, use_fast=False)
    ds = bundle.dev_dataset if args.split == "dev" else bundle.train_dataset if args.split == "train" else bundle.test_dataset

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
        num_workers=0,
    )

    # load model
    model_cls = resolve_single_model_class()
    model = instantiate_single_model(model_cls, ckpt=args.ckpt, num_aspects=len(ASPECTS)).to(device)
    load_single_checkpoint(model, ckpt_path)

    y_true, y_prob = collect_probs_and_labels(model, loader, device, num_aspects=len(ASPECTS))

    # sweep thresholds
    thresholds = np.arange(args.thr_start, args.thr_end + 1e-9, args.thr_step)
    macro_f1s = []
    micro_f1s = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int32)
        macro_f1s.append(compute_macro_f1(y_true, y_pred))
        micro_f1s.append(compute_micro_f1(y_true, y_pred))

    # output folder
    out_dir = Path(args.out_dir) if args.out_dir.strip() else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # save csv
    csv_path = out_dir / f"f1_vs_threshold_{args.split}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("threshold,macro_f1,micro_f1\n")
        for thr, mf, mif in zip(thresholds, macro_f1s, micro_f1s):
            f.write(f"{thr:.2f},{mf:.6f},{mif:.6f}\n")

    # plot (Figure 5)
    fig_path = out_dir / "figure5_f1_vs_threshold.png"
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(thresholds, macro_f1s, label="Macro-F1")
    plt.plot(thresholds, micro_f1s, label="Micro-F1")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Threshold (Validation/Dev set)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

    # also save a PDF if you want high-quality vector figure
    pdf_path = out_dir / "figure5_f1_vs_threshold.pdf"
    plt.savefig(pdf_path)

    # print best threshold by macro f1
    best_idx = int(np.argmax(macro_f1s))
    print(f"[OK] Saved CSV : {csv_path}")
    print(f"[OK] Saved PNG : {fig_path}")
    print(f"[OK] Saved PDF : {pdf_path}")
    print(f"[BEST] by Macro-F1: thr={thresholds[best_idx]:.2f}, macro_f1={macro_f1s[best_idx]:.4f}, micro_f1={micro_f1s[best_idx]:.4f}")


if __name__ == "__main__":
    main()