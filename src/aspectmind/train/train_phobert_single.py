from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch
from aspectmind.models.phobert_single import PhoBERTSingleTask, ASPECTS


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_on_loader(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_pred = []

    for batch in loader:
        # move to device
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            labels=None,
        )
        logits = out["logits"]  # (B, 6)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).long()

        all_y.append(batch["labels"].long().cpu())
        all_pred.append(pred.cpu())

    y = torch.cat(all_y, dim=0).numpy()
    p = torch.cat(all_pred, dim=0).numpy()

    # metrics
    # avoid importing sklearn in training loop? but you already have it; use it for consistent F1
    from sklearn.metrics import f1_score

    macro = f1_score(y, p, average="macro", zero_division=0)
    micro = f1_score(y, p, average="micro", zero_division=0)
    per_aspect = f1_score(y, p, average=None, zero_division=0)

    metrics = {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
    }
    for a, f1 in zip(ASPECTS, per_aspect):
        metrics[f"f1_{a}"] = float(f1)
    return metrics


def main():
    # =====================
    # Config (simple & stable)
    # =====================
    seed = 42
    model_name = "vinai/phobert-base"
    max_length = 256
    use_fast = False

    batch_size = 8
    num_epochs = 3
    lr = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    grad_clip = 1.0

    use_amp = True  # mixed precision on GPU

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =====================
    # Data
    # =====================
    bundle = build_phobert_datasets(model_name=model_name, max_length=max_length, use_fast=use_fast)

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
    )
    dev_loader = DataLoader(
        bundle.dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
    )

    # =====================
    # Model
    # =====================
    model = PhoBERTSingleTask(model_name=model_name, dropout=0.1)
    model.to(device)

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # =====================
    # Run dir
    # =====================
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path("runs") / f"phobert_single_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = {
        "seed": seed,
        "model_name": model_name,
        "max_length": max_length,
        "use_fast": use_fast,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "grad_clip": grad_clip,
        "use_amp": use_amp,
        "aspects": ASPECTS,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # =====================
    # Train loop
    # =====================
    best_macro = -1.0
    best_path = out_dir / "best_model.pt"

    print("\nTraining PhoBERT single-task detection...")
    print("Train steps/epoch:", len(train_loader))
    print("Total steps:", total_steps, "| Warmup:", warmup_steps)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            global_step += 1

            # move to device
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids", None),
                    labels=batch["labels"],
                )
                loss = out["loss"]

            scaler.scale(loss).backward()

            # grad clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())

            if global_step % 50 == 0:
                avg = running_loss / 50
                running_loss = 0.0
                lr_now = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | step {global_step}/{total_steps} | loss {avg:.4f} | lr {lr_now:.2e}")

        # ===== Eval end of epoch
        metrics = eval_on_loader(model, dev_loader, device)
        print(f"\n[DEV] Epoch {epoch}: macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f}")
        for a in ASPECTS:
            print(f"  - {a}: {metrics[f'f1_{a}']:.4f}")

        # Save epoch metrics
        (out_dir / f"metrics_epoch_{epoch}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Save best
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_macro_f1": best_macro,
                },
                best_path,
            )
            print(f"✅ Saved BEST model to: {best_path} (macro_f1={best_macro:.4f})")

    # Final summary
    summary = {"best_macro_f1": float(best_macro), "best_model_path": str(best_path.as_posix())}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDONE.")
    print("Run dir:", out_dir.resolve())
    print("Best macro_f1:", best_macro)
    print("Best model:", best_path.resolve())


if __name__ == "__main__":
    main()
