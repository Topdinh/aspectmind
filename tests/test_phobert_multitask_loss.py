import torch
from torch.utils.data import DataLoader

from aspectmind.models.phobert_multitask import PhoBERTMultiTask
from aspectmind.data.dataset_phobert_multitask import (
    PhoBERTMultiTaskDataset,
    collate_multitask,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PhoBERTMultiTaskDataset("train", max_length=128)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_multitask)

    model = PhoBERTMultiTask()
    model.to(device)
    model.train()

    found = False

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # kiểm tra có sentiment label trong batch không
        if batch["sent_mask"].sum().item() == 0:
            continue

        found = True

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss, loss_aspect, loss_sent, sent_count = model.compute_loss(
            outputs,
            y_aspect=batch["y_aspect"],
            y_sent=batch["y_sent"],
            sent_mask=batch["sent_mask"],
        )

        print("loss:", float(loss))
        print("loss_aspect:", float(loss_aspect))
        print("loss_sent:", float(loss_sent))
        print("sent_count:", float(sent_count))
        break

    if not found:
        print("ERROR: No batch in dataset contains sentiment labels (sent_mask > 0).")


if __name__ == "__main__":
    main()
