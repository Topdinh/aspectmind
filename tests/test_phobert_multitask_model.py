import torch
from torch.utils.data import DataLoader

from aspectmind.data.dataset_phobert_multitask import (
    PhoBERTMultiTaskDataset,
    collate_multitask,
)
from aspectmind.models.phobert_multitask import PhoBERTMultiTask


def main():
    ds = PhoBERTMultiTaskDataset(split="train", max_length=64)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_multitask)

    batch = next(iter(dl))

    model = PhoBERTMultiTask()
    model.eval()

    with torch.no_grad():
        aspect_logits, sent_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    print("Aspect logits:", aspect_logits.shape)
    print("Sent logits:", sent_logits.shape)


if __name__ == "__main__":
    main()
