from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from aspectmind.data.dataset_phobert_multitask import PhoBERTMultiTaskDataset, collate_multitask, ASPECTS
from aspectmind.models.phobert_multitask import PhoBERTMultiTask


@dataclass
class TrainConfig:
    ckpt: str = "vinai/phobert-base"
    max_length: int = 128
    batch_size: int = 8
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    w_aspect: float = 1.0
    w_sent: float = 1.0
    threshold: float = 0.5
    num_workers: int = 0
    seed: int = 42


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: PhoBERTMultiTask, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Dict[str, Any]:
    model.eval()

    # Accumulate predictions for aspect metrics
    all_pred = []
    all_true = []

    # For sentiment (masked): accuracy over sentiment slots only
    sent_correct = 0
    sent_total = 0

    for batch in loader:
        # move
        batch = {k: v.to(device) for k, v in batch.items()}

        aspect_logits, sent_logits = model(batch["input_ids"], batch["attention_mask"])
        aspect_prob = torch.sigmoid(aspect_logits)
        aspect_pred = (aspect_prob >= cfg.threshold).long()
        aspect_true = batch["y_aspect"].long()

        all_pred.append(aspect_pred.cpu())
        all_true.append(aspect_true.cpu())

        # Sentiment: only where sent_mask==1
        mask = batch["sent_mask"]  # (B, A)
        if mask.sum() > 0:
            sent_pred = sent_logits.argmax(dim=-1)  # (B, A)
            sent_true = batch["y_sent"]            # (B, A)
            m = mask.bool()
            sent_correct += (sent_pred[m] == sent_true[m]).sum().item()
            sent_total += m.sum().item()

    y_pred = torch.cat(all_pred, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy()

    # sklearn-style metrics without importing sklearn (keep deps minimal):
    # macro-F1 over aspects
    eps = 1e-9
    per_aspect = {}
    f1s = []

    for i, a in enumerate(ASPECTS):
        tp = ((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()
        fp = ((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        per_aspect[a] = float(f1)
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / len(f1s))
    micro_tp = ((y_pred == 1) & (y_true == 1)).sum()
    micro_fp = ((y_pred == 1) & (y_true == 0)).sum()
    micro_fn = ((y_pred == 0) & (y_true == 1)).sum()
    micro_p = micro_tp / (micro_tp + micro_fp + eps)
    micro_r = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r + eps))

    sent_acc = float(sent_correct / sent_total) if sent_total > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_aspect_f1": per_aspect,
        "sent_acc_masked": sent_acc,
        "sent_total": int(sent_total),
    }


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path("runs") / f"phobert_multitask_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = PhoBERTMultiTaskDataset("train", ckpt=cfg.ckpt, max_length=cfg.max_length)
    dev_ds = PhoBERTMultiTaskDataset("dev", ckpt=cfg.ckpt, max_length=cfg.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_multitask,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_multitask,
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Dev size: {len(dev_ds)}")

    # Model
    model = PhoBERTMultiTask(ckpt=cfg.ckpt).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"Train steps/epoch: {len(train_loader)}")
    print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")
    print("Training PhoBERT multi-task (aspect + sentiment)...")

    best_macro = -1.0
    best_path = run_dir / "best_model.pt"

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, loss_aspect, loss_sent, sent_count = model.compute_loss(
                batch,
                w_aspect=cfg.w_aspect,
                w_sent=cfg.w_sent,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch} | step {step}/{total_steps} "
                    f"| loss {loss.item():.4f} (asp {loss_aspect.item():.4f}, sent {loss_sent.item():.4f}, sent_n={sent_count:.0f}) "
                    f"| lr {lr_now:.2e}"
                )

        # Eval dev
        metrics = evaluate(model, dev_loader, device, cfg)
        print(f"\n[DEV] Epoch {epoch}: macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} sent_acc={metrics['sent_acc_masked']:.4f} (n={metrics['sent_total']})")
        for a, f1 in metrics["per_aspect_f1"].items():
            print(f"  - {a}: {f1:.4f}")

        # Save best by macro_f1 (aspect detection)
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "metrics_dev": metrics,
                },
                best_path,
            )
            print(f"✅ Saved BEST model to: {best_path} (macro_f1={best_macro:.4f})")

        # Save metrics per epoch
        (run_dir / f"metrics_dev_epoch{epoch}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDONE.")
    print(f"Run dir: {run_dir}")
    print(f"Best macro_f1: {best_macro}")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
