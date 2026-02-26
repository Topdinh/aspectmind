from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class MultiTaskLossOutput:
    loss: torch.Tensor
    loss_aspect: torch.Tensor
    loss_sent: torch.Tensor
    sent_count: torch.Tensor  # number of active sentiment positions (sum of mask)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss:
      - Aspect: BCEWithLogits
      - Sentiment: CrossEntropy, masked by sent_mask
    """

    def __init__(self, lambda_sent: float = 1.0):
        super().__init__()
        self.lambda_sent = float(lambda_sent)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        # We'll do CE manually with reduction='none' to apply mask
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        aspect_logits: torch.Tensor,  # (B, A)
        sent_logits: torch.Tensor,    # (B, A, C)
        y_aspect: torch.Tensor,       # (B, A) float
        y_sent: torch.Tensor,         # (B, A) long
        sent_mask: torch.Tensor,      # (B, A) float {0,1}
    ) -> MultiTaskLossOutput:
        # Aspect loss
        loss_aspect = self.bce(aspect_logits, y_aspect)

        # Sentiment loss (masked)
        B, A, C = sent_logits.shape
        sent_logits_flat = sent_logits.reshape(B * A, C)     # (B*A, C)
        y_sent_flat = y_sent.reshape(B * A)                  # (B*A,)
        mask_flat = sent_mask.reshape(B * A).float()         # (B*A,)

        ce_per = self.ce(sent_logits_flat, y_sent_flat)      # (B*A,)
        masked = ce_per * mask_flat                          # (B*A,)

        sent_count = mask_flat.sum().clamp(min=1.0)          # avoid div0
        loss_sent = masked.sum() / sent_count

        loss = loss_aspect + self.lambda_sent * loss_sent

        return MultiTaskLossOutput(
            loss=loss,
            loss_aspect=loss_aspect.detach(),
            loss_sent=loss_sent.detach(),
            sent_count=sent_count.detach(),
        )
