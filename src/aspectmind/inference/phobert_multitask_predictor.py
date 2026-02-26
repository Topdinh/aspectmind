from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from aspectmind.models.phobert_multitask import PhoBERTMultiTask

ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]
ID2SENT = {0: "pos", 1: "neg", 2: "neu"}
SENT2ID = {"pos": 0, "neg": 1, "neu": 2}


@dataclass
class MultiTaskOutput:
    pred_aspect: Dict[str, int]
    proba_aspect: Dict[str, float]
    sent: Dict[str, Optional[str]]  # None nếu aspect không xuất hiện
    sent_proba: Dict[str, Optional[Dict[str, float]]]  # None nếu aspect không xuất hiện


class PhoBERTMultiTaskPredictor:
    """
    Inference wrapper for PhoBERT multi-task:
      - Aspect detection: multi-label (sigmoid)
      - Sentiment: per-aspect 3-class (softmax) computed only when aspect is predicted

    It loads:
      - base ckpt: vinai/phobert-base (tokenizer + encoder)
      - fine-tuned head weights from best_model.pt saved by your training script

    Notes:
      - best_model.pt may be either:
          * a raw torch state_dict
          * a checkpoint dict containing "model_state_dict" (your training script)
          * a checkpoint dict containing "state_dict" (other variants)
      - This predictor does NOT change any existing project behavior; it's a new module.
    """

    def __init__(
        self,
        run_dir: str,
        ckpt: str = "vinai/phobert-base",
        max_length: int = 256,
        aspect_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.run_dir = run_dir
        self.ckpt = ckpt
        self.max_length = int(max_length)
        self.aspect_threshold = float(aspect_threshold)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt, use_fast=False)

        self.model = PhoBERTMultiTask(
            ckpt=self.ckpt,
            num_aspects=len(ASPECTS),
            num_sent_classes=3,
        ).to(self.device)

        weights_path = Path(self.run_dir) / "best_model.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        state = torch.load(str(weights_path), map_location="cpu")

        # --- FIX: robustly pick the real state_dict from various checkpoint formats ---
        # Your training script saves:
        #   {"model_state_dict": ..., "cfg": ..., "metrics_dev": ...}
        # Other variants may save:
        #   {"state_dict": ...}
        if isinstance(state, dict):
            if isinstance(state.get("model_state_dict"), dict):
                state = state["model_state_dict"]
            elif isinstance(state.get("state_dict"), dict):
                state = state["state_dict"]

        if not isinstance(state, dict):
            raise TypeError(
                f"Checkpoint format not understood. Expected dict state_dict, got: {type(state)}"
            )

        # Strip common prefixes if present (keep behavior robust)
        # e.g., "model.encoder..." -> "encoder..."
        def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            if not sd:
                return sd
            if all(k.startswith(prefix) for k in sd.keys()):
                return {k[len(prefix):]: v for k, v in sd.items()}
            return sd

        state = _strip_prefix(state, "model.")
        state = _strip_prefix(state, "module.")
        # ---------------------------------------------------------------------------

        missing, unexpected = self.model.load_state_dict(state, strict=False)

        # strict=False is intentional (robust across minor changes)
        if missing:
            print(f"[PhoBERTMultiTaskPredictor] Missing keys: {missing}")
        if unexpected:
            print(f"[PhoBERTMultiTaskPredictor] Unexpected keys: {unexpected}")

        self.model.eval()

    @torch.no_grad()
    def _forward(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        aspect_logits, sent_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # aspect_logits: (1, 6)
        # sent_logits:   (1, 6, 3)
        return aspect_logits.squeeze(0), sent_logits.squeeze(0)

    def predict(self, text: str) -> Dict[str, int]:
        aspect_logits, _ = self._forward(text)
        probs = torch.sigmoid(aspect_logits)
        pred = (probs >= self.aspect_threshold).long().tolist()
        return {a: int(pred[i]) for i, a in enumerate(ASPECTS)}

    def predict_proba(self, text: str) -> Dict[str, float]:
        aspect_logits, _ = self._forward(text)
        probs = torch.sigmoid(aspect_logits).tolist()
        return {a: float(probs[i]) for i, a in enumerate(ASPECTS)}

    def predict_with_sentiment(self, text: str) -> MultiTaskOutput:
        aspect_logits, sent_logits = self._forward(text)

        aspect_probs = torch.sigmoid(aspect_logits)  # (6,)
        aspect_pred = (aspect_probs >= self.aspect_threshold).long()  # (6,)

        pred_aspect = {a: int(aspect_pred[i].item()) for i, a in enumerate(ASPECTS)}
        proba_aspect = {a: float(aspect_probs[i].item()) for i, a in enumerate(ASPECTS)}

        sent: Dict[str, Optional[str]] = {}
        sent_proba: Dict[str, Optional[Dict[str, float]]] = {}

        # sentiment probs per aspect: (6,3)
        sent_probs = F.softmax(sent_logits, dim=-1)

        for i, a in enumerate(ASPECTS):
            if pred_aspect[a] == 0:
                sent[a] = None
                sent_proba[a] = None
                continue

            p = sent_probs[i]  # (3,)
            sid = int(torch.argmax(p).item())
            sent[a] = ID2SENT.get(sid, "neu")

            sent_proba[a] = {
                "pos": float(p[SENT2ID["pos"]].item()),
                "neg": float(p[SENT2ID["neg"]].item()),
                "neu": float(p[SENT2ID["neu"]].item()),
            }

        return MultiTaskOutput(
            pred_aspect=pred_aspect,
            proba_aspect=proba_aspect,
            sent=sent,
            sent_proba=sent_proba,
        )
