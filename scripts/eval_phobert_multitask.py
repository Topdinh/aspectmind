# scripts/eval_phobert_multitask.py
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from aspectmind.data.dataset_phobert_multitask import PhoBERTMultiTaskDataset, collate_multitask
from aspectmind.models.phobert_multitask import PhoBERTMultiTask

ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]
ID2SENT = {0: "pos", 1: "neg", 2: "neu"}


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


@torch.no_grad()
def evaluate(
    model: PhoBERTMultiTask,
    loader: DataLoader,
    device: torch.device,
    aspect_threshold: float = 0.5,
) -> Dict:
    model.eval()

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    # sentiment
    sent_correct = 0
    sent_total = 0

    # for per-aspect F1
    y_true_by_aspect = {a: [] for a in ASPECTS}
    y_pred_by_aspect = {a: [] for a in ASPECTS}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y_aspect = batch["y_aspect"].to(device)          # (B,6) float 0/1
        y_sent = batch["y_sent"].to(device)              # (B,6) long
        sent_mask = batch["sent_mask"].to(device)        # (B,6) float 0/1

        aspect_logits, sent_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        # aspect_logits: (B,6)
        # sent_logits: (B,6,3)

        aspect_probs = _sigmoid(aspect_logits)
        aspect_pred = (aspect_probs >= aspect_threshold).long()  # (B,6)
        aspect_true = y_aspect.long()  # (B,6)

        # aspect metrics collect
        y_true_all.append(aspect_true.detach().cpu().numpy())
        y_pred_all.append(aspect_pred.detach().cpu().numpy())

        # per-aspect collect
        for i, a in enumerate(ASPECTS):
            y_true_by_aspect[a].append(aspect_true[:, i].detach().cpu().numpy())
            y_pred_by_aspect[a].append(aspect_pred[:, i].detach().cpu().numpy())

        # sentiment accuracy (only where sent_mask==1)
        # prediction: argmax over 3
        sent_pred = torch.argmax(sent_logits, dim=-1)  # (B,6)

        mask = sent_mask > 0.0  # bool (B,6)
        if mask.any():
            correct = (sent_pred[mask] == y_sent[mask]).sum().item()
            total = int(mask.sum().item())
            sent_correct += int(correct)
            sent_total += total

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_aspect = {}
    for a in ASPECTS:
        yt = np.concatenate(y_true_by_aspect[a], axis=0)
        yp = np.concatenate(y_pred_by_aspect[a], axis=0)
        per_aspect[a] = float(f1_score(yt, yp, average="binary", zero_division=0))

    sent_acc = float(sent_correct / sent_total) if sent_total > 0 else 0.0

    return {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_aspect_f1": per_aspect,
        "sent_acc": sent_acc,
        "sent_n": int(sent_total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default="runs/phobert_multitask_2026-02-15_23-44-18",
        help="Folder chứa best_model.pt",
    )
    ap.add_argument("--ckpt", type=str, default="vinai/phobert-base")
    ap.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # dataset/loader
    ds = PhoBERTMultiTaskDataset(split=args.split, ckpt=args.ckpt, max_length=args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multitask)

    # model
    model = PhoBERTMultiTask(ckpt=args.ckpt, num_aspects=len(ASPECTS), num_sent_classes=3).to(device)

    weights_path = f"{args.run_dir}/best_model.pt"
    state = torch.load(weights_path, map_location="cpu")

    # allow either:
    #  - raw state_dict
    #  - dict with "state_dict"
    #  - dict with "model_state_dict" (bạn từng gặp case này)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[eval] Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[eval] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    metrics = evaluate(model, loader, device=device, aspect_threshold=args.threshold)

    print(f"\n[EVAL:{args.split.upper()}] run_dir={args.run_dir}")
    print(f"  macro_f1 : {metrics['macro_f1']:.4f}")
    print(f"  micro_f1 : {metrics['micro_f1']:.4f}")
    print(f"  sent_acc : {metrics['sent_acc']:.4f} (n={metrics['sent_n']})")
    print("  per-aspect F1:")
    for a in ASPECTS:
        print(f"    - {a:11s}: {metrics['per_aspect_f1'][a]:.4f}")


if __name__ == "__main__":
    main()
