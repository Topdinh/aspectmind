from torch.utils.data import DataLoader

from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch


bundle = build_phobert_datasets(model_name="vinai/phobert-base", max_length=128, use_fast=False)

dl = DataLoader(
    bundle.train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
)

batch = next(iter(dl))
print("Keys:", batch.keys())
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("labels:", batch["labels"].shape)
print("labels sample:", batch["labels"][0].tolist())
