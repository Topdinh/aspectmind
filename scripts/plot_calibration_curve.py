from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch
from aspectmind.models.phobert_single import PhoBERTSingleTask


ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


def load_temperature(temp_json: Path) -> float:
    data = json.loads(temp_json.read_text(encoding="utf-8"))
    return float(data["temperature"])


def load_single_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt)}")

    state = None
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt

    if state is None:
        raise KeyError("Checkpoint does not contain model_state_dict/state_dict")

    if any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model.") :]: v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    return ckpt


@torch.no_grad()
def collect_logits_and_labels(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, num_aspects: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_y = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        if not (isinstance(out, dict) and "logits" in out):
            raise TypeError("Model output must be dict with key 'logits'")
        logits = out["logits"]
        if logits.dim() != 2 or logits.size(-1) != num_aspects:
            raise ValueError(f"Unexpected logits shape {tuple(logits.shape)} expected (N,{num_aspects})")

        y = batch["labels"].to(device).float()
        y = torch.clamp(y, 0.0, 1.0)

        all_logits.append(logits)
        all_y.append(y)

    return torch.cat(all_logits, dim=0), torch.cat(all_y, dim=0)


def reliability_curve_binary_flat(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """
    probs, y: flattened arrays (N_total,)
    Return:
      bin_centers, avg_confidence, avg_accuracy
    """
    probs = np.clip(probs, 0.0, 1.0)
    y = (y > 0.5).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    avg_conf = np.zeros(n_bins, dtype=np.float64)
    avg_acc = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] > 0:
            avg_conf[b] = float(probs[m].mean())
            avg_acc[b] = float(y[m].mean())

    # Use bin centers for x-axis
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    return bin_centers, avg_conf, avg_acc, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--temperature_json", default="", help="Path to temperature_dev.json")
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="vinai/phobert-base")
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--out_dir", type=str, default="", help="default: same folder as ckpt")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    # resolve temperature json
    if args.temperature_json.strip():
        temp_json = Path(args.temperature_json)
    else:
        temp_json = ckpt_path.parent / "temperature_dev.json"

    if not temp_json.exists():
        raise FileNotFoundError(f"temperature json not found: {temp_json}")

    T = load_temperature(temp_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset bundle (DEV)
    bundle = build_phobert_datasets(model_name=args.ckpt, max_length=args.max_length, use_fast=False)
    ds = bundle.dev_dataset

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
        num_workers=0,
    )

    # Load model
    model = PhoBERTSingleTask(model_name=args.ckpt, dropout=0.1).to(device)
    load_single_checkpoint(model, ckpt_path)

    logits, y = collect_logits_and_labels(model, loader, device, num_aspects=len(ASPECTS))

    # Before / After probabilities
    probs_before = torch.sigmoid(logits).cpu().numpy()
    probs_after = torch.sigmoid(logits / max(T, 1e-6)).cpu().numpy()
    y_np = y.cpu().numpy()

    # Flatten across aspects => reliability curve overall
    p_b = probs_before.reshape(-1)
    p_a = probs_after.reshape(-1)
    y_f = y_np.reshape(-1)

    x, conf_b, acc_b, cnt_b = reliability_curve_binary_flat(p_b, y_f, n_bins=args.ece_bins)
    _, conf_a, acc_a, cnt_a = reliability_curve_binary_flat(p_a, y_f, n_bins=args.ece_bins)

    out_dir = Path(args.out_dir) if args.out_dir.strip() else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot (Figure 6)
    png_path = out_dir / "figure6_calibration_curve.png"
    pdf_path = out_dir / "figure6_calibration_curve.pdf"

    plt.figure(figsize=(7.4, 5.2))
    # perfect calibration diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    # Use avg confidence (x) vs avg accuracy (y)
    # (We plot acc vs conf to show calibration; points with 0 counts will appear at 0,0 -> mask them)
    m_b = cnt_b > 0
    m_a = cnt_a > 0
    plt.plot(conf_b[m_b], acc_b[m_b], marker="o", linewidth=1.5, label="Before (T=1.0)")
    plt.plot(conf_a[m_a], acc_a[m_a], marker="o", linewidth=1.5, label=f"After (T={T:.3f})")

    plt.xlabel("Predicted probability (confidence)")
    plt.ylabel("True frequency (accuracy)")
    plt.title("Calibration Curve (Dev set)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")
    print(f"[INFO] Temperature T = {T:.6f} (from {temp_json})")


if __name__ == "__main__":
    main()