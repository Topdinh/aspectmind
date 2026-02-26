from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


class PhoBERTSingleTask(nn.Module):
    """
    PhoBERT-based multi-label classifier for aspect detection (6 labels).
    Output logits shape: (batch_size, 6)

    Temperature scaling support (calibration):
      - This module stores a scalar temperature T (default 1.0).
      - Forward() behavior is NOT changed: it returns raw logits (unscaled).
      - Use `scale_logits(logits)` (or `scale_logits(logits, T=...)`) externally for calibration.
      - This keeps training/eval code backward-compatible and avoids affecting training loss.
    """

    def __init__(self, model_name: str = "vinai/phobert-base", dropout: float = 0.1):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(ASPECTS))

        # ✅ Temperature scaling (calibration)
        # Register as buffer so it:
        # - moves with .to(device)
        # - is saved/loaded with state_dict
        # - is NOT optimized during normal training unless you explicitly do so
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    # ----------------------------
    # Temperature scaling helpers
    # ----------------------------
    def set_temperature(self, T: float) -> None:
        """
        Set temperature for calibration. Must be > 0.
        This does NOT change forward(); you must use scale_logits() externally.
        """
        T_val = float(T)
        if not (T_val > 0.0):
            raise ValueError(f"Temperature must be > 0, got {T}")
        # keep same device/dtype
        self.temperature.data = torch.tensor(T_val, device=self.temperature.device, dtype=self.temperature.dtype)

    def get_temperature(self) -> float:
        """Return current stored temperature as Python float."""
        return float(self.temperature.detach().cpu().item())

    def scale_logits(self, logits: torch.Tensor, T: float | None = None) -> torch.Tensor:
        """
        Return calibrated logits = logits / T.

        If T is None -> uses self.temperature.
        Safety: clamps T to avoid division by ~0.
        """
        if T is None:
            T_t = self.temperature
        else:
            T_val = float(T)
            if not (T_val > 0.0):
                raise ValueError(f"Temperature must be > 0, got {T}")
            T_t = torch.tensor(T_val, device=logits.device, dtype=logits.dtype)

        # numerical safety
        T_t = torch.clamp(T_t, min=1e-6)
        return logits / T_t

    # ----------------------------
    # Forward (unchanged behavior)
    # ----------------------------
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS representation
        pooled = outputs.last_hidden_state[:, 0]  # (B, hidden)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)  # (B, 6)

        loss = None
        if labels is not None:
            # multi-label -> BCEWithLogitsLoss (use RAW logits, not temperature-scaled)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }