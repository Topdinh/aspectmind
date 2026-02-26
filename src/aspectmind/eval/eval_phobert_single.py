from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from aspectmind.data.loader import load_config, load_split
from aspectmind.data.transformer_dataset import collate_batch
from aspectmind.models.phobert_single import PhoBERTSingleTask, ASPECTS


def labels_to_multihot(label_dict: Dict[str, str]) -> np.ndarray:
    y = np.zeros(len(ASPECTS), dtype=np.int64)
    for i, a in enumerate(ASPECTS):
        if label_dict.get(a, "not_mentioned") != "not_mentioned":
            y[i] = 1
    return y


def prepare_samples_for_eval(samples: List[Dict]) -> List[Dict]:
    # keep same structure used by dataset
    return samples


@torch.no_grad()
def eval_model(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_pred = []

    for batch in loader:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            labels=None,
        )
        probs = torch.sigmoid(out["logits"])
        pred = (probs >= 0.5).long()

        all_y.append(batch["labels"].long().cpu())
        all_pred.append(pred.cpu())

    y = torch.cat(all_y, dim=0).numpy()
    p = torch.cat(all_pred, dim=0).numpy()

    from sklearn.metrics import f1_score, classification_report

    macro = f1_score(y, p, average="macro", zero_division=0)
    micro = f1_score(y, p, average="micro", zero_division=0)
    per_aspect = f1_score(y, p, average=None, zero_division=0)

    print("\n=== PHOBERT SINGLE METRICS (TEST) ===")
    print(f"Macro-F1: {macro:.4f}")
    print(f"Micro-F1: {micro:.4f}")
    print("\nPer-aspect F1:")
    for a, f1 in zip(ASPECTS, per_aspect):
        print(f" - {a}: {f1:.4f}")

    print("\nClassification report (per label):")
    print(classification_report(y, p, target_names=ASPECTS, zero_division=0))

    metrics = {
        "split": "test",
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_aspect_f1": {a: float(f1) for a, f1 in zip(ASPECTS, per_aspect)},
        "aspects": ASPECTS,
    }
    return metrics


def main():
    # IMPORTANT: set this to your run dir
    run_dir = Path("runs/phobert_single_2026-02-15_16-15-19")
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model_name = cfg["model_name"]
    max_length = cfg["max_length"]
    use_fast = cfg.get("use_fast", False)

    # Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    # Load test samples
    cfg_map = load_config()
    test_path = Path("data/raw/uit_visd4sa/data/test.jsonl")
    test_samples, _ = load_split(
        test_path,
        cfg_map["aspect_mapping"],
        cfg_map["sentiment_mapping"],
        set(cfg_map.get("ignored_aspects", [])),
    )

    # Build dataset on-the-fly (reuse same dataset class)
    from aspectmind.data.transformer_dataset import PhoBERTDetectionDataset

    test_ds = PhoBERTDetectionDataset(test_samples, tokenizer=tokenizer, max_length=max_length)

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(tokenizer, b),
    )

    # Build model + load weights
    model = PhoBERTSingleTask(model_name=model_name, dropout=0.1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics = eval_model(model, test_loader, device)

    out_path = run_dir / "metrics_test.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved test metrics to: {out_path}")


if __name__ == "__main__":
    main()
