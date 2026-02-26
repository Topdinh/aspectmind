from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class PhoBERTMultiTask(nn.Module):
    def __init__(
        self,
        ckpt: str = "vinai/phobert-base",
        num_aspects: int = 6,
        num_sent_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(ckpt)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Head 1: Aspect detection (multi-label)
        self.aspect_head = nn.Linear(hidden_size, num_aspects)

        # Head 2: Sentiment classification (per aspect)
        self.sent_head = nn.Linear(hidden_size, num_aspects * num_sent_classes)

        self.num_aspects = num_aspects
        self.num_sent_classes = num_sent_classes

        # losses
        self._bce = nn.BCEWithLogitsLoss()
        self._ce_none = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)

        # Aspect logits
        aspect_logits = self.aspect_head(cls)  # (B, A)

        # Sentiment logits
        sent_logits = self.sent_head(cls)  # (B, A*C)
        sent_logits = sent_logits.view(-1, self.num_aspects, self.num_sent_classes)  # (B, A, C)

        return aspect_logits, sent_logits

    def compute_loss(
        self,
        batch: object | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        y_aspect: torch.Tensor | None = None,
        y_sent: torch.Tensor | None = None,
        sent_mask: torch.Tensor | None = None,
        w_aspect: float = 1.0,
        w_sent: float = 1.0,
        **kwargs,
    ):
        """
        Supports multiple call styles:

        1) model.compute_loss(batch_dict)
        2) model.compute_loss(batch=batch_dict)
        3) model.compute_loss(input_ids=..., attention_mask=..., y_aspect=..., y_sent=..., sent_mask=...)
        4) model.compute_loss((aspect_logits, sent_logits), y_aspect=..., y_sent=..., sent_mask=...)
        5) model.compute_loss((input_ids, attention_mask), y_aspect=..., y_sent=..., sent_mask=...)

        Returns:
          total_loss, loss_aspect, loss_sent, sent_count
        """

        # Allow batch passed via kwargs
        if batch is None and "batch" in kwargs:
            batch = kwargs["batch"]

        aspect_logits = None
        sent_logits = None

        # --- CASE: batch is a tuple/list of 2 tensors ---
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and all(isinstance(x, torch.Tensor) for x in batch):
            t0, t1 = batch[0], batch[1]

            # Case 4: logits tuple: (B,A) and (B,A,C)
            if t0.dim() == 2 and t1.dim() == 3 and t0.size(1) == self.num_aspects and t1.size(1) == self.num_aspects:
                aspect_logits, sent_logits = t0, t1
                batch = None  # labels must come from args/kwargs
            # Case 5: minimal batch tuple: (input_ids, attention_mask)
            elif t0.dim() == 2 and t1.dim() == 2 and t0.shape == t1.shape:
                # heuristic: input_ids usually long/int, attention_mask usually long/int
                input_ids = t0
                attention_mask = t1
                batch = None
            else:
                raise TypeError(
                    "compute_loss got a tuple/list of 2 tensors but cannot infer whether it is "
                    "(aspect_logits, sent_logits) or (input_ids, attention_mask). "
                    f"Got shapes: {tuple(t0.shape)} and {tuple(t1.shape)}"
                )

        # --- CASE: batch is dict ---
        if isinstance(batch, dict):
            # allow overrides from kwargs
            for k in ["input_ids", "attention_mask", "y_aspect", "y_sent", "sent_mask"]:
                if k in kwargs:
                    batch[k] = kwargs[k]
            input_ids = batch.get("input_ids", input_ids)
            attention_mask = batch.get("attention_mask", attention_mask)
            y_aspect = batch.get("y_aspect", y_aspect)
            y_sent = batch.get("y_sent", y_sent)
            sent_mask = batch.get("sent_mask", sent_mask)
        else:
            # if not dict, still accept explicit tensors via kwargs
            if "input_ids" in kwargs:
                input_ids = kwargs["input_ids"]
            if "attention_mask" in kwargs:
                attention_mask = kwargs["attention_mask"]
            if "y_aspect" in kwargs:
                y_aspect = kwargs["y_aspect"]
            if "y_sent" in kwargs:
                y_sent = kwargs["y_sent"]
            if "sent_mask" in kwargs:
                sent_mask = kwargs["sent_mask"]

        # If logits were NOT provided, we must forward
        if aspect_logits is None or sent_logits is None:
            if input_ids is None or attention_mask is None:
                raise ValueError("Missing input_ids/attention_mask (or logits tuple) for compute_loss()")

            aspect_logits, sent_logits = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        device = aspect_logits.device

        # Labels must exist for aspect loss
        if y_aspect is None:
            raise ValueError("Missing y_aspect for compute_loss()")
        y_aspect = y_aspect.to(device)  # (B,A) float

        # Sent labels optional (allow aspect-only)
        if y_sent is None:
            y_sent = torch.zeros_like(y_aspect, dtype=torch.long, device=device)
        else:
            y_sent = y_sent.to(device)  # (B,A) long

        if sent_mask is None:
            sent_mask = torch.zeros_like(y_aspect, dtype=torch.float32, device=device)
        else:
            sent_mask = sent_mask.to(device)  # (B,A) float

        # 1) Aspect loss
        loss_aspect = self._bce(aspect_logits, y_aspect)

        # 2) Sentiment loss (masked)
        B, A, C = sent_logits.shape
        sent_logits_flat = sent_logits.reshape(B * A, C)
        y_sent_flat = y_sent.reshape(B * A)
        mask_flat = sent_mask.reshape(B * A)

        loss_each = self._ce_none(sent_logits_flat, y_sent_flat)  # (B*A,)
        masked = loss_each * mask_flat

        sent_count_t = mask_flat.sum()
        sent_count = float(sent_count_t.item())
        denom = sent_count_t.clamp(min=1.0)
        loss_sent = masked.sum() / denom

        total_loss = (w_aspect * loss_aspect) + (w_sent * loss_sent)

        return total_loss, loss_aspect, loss_sent, sent_count
