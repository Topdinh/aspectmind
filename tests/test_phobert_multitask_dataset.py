import torch
from torch.utils.data import DataLoader

from aspectmind.data.dataset_phobert_multitask import PhoBERTMultiTaskDataset, collate_multitask


def main():
    ds = PhoBERTMultiTaskDataset(split="train", max_length=128)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_multitask)

    batch = next(iter(dl))
    print("Keys:", batch.keys())
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("y_aspect:", batch["y_aspect"].shape, batch["y_aspect"].dtype)
    print("y_sent:", batch["y_sent"].shape, batch["y_sent"].dtype)
    print("sent_mask:", batch["sent_mask"].shape, batch["sent_mask"].dtype)

    # sanity: mask=0 => aspect should be 0
    ok = torch.all((batch["sent_mask"] == 1.0) == (batch["y_aspect"] == 1.0))
    print("Mask-aspect consistency:", bool(ok))


if __name__ == "__main__":
    main()
