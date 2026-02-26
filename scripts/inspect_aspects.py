import json
from collections import Counter

counter = Counter()

with open("data/raw/uit_visd4sa/data/train.jsonl", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        for lbl in obj["labels"]:
            aspect = lbl[2].split("#")[0]
            counter[aspect] += 1

print("Aspect frequency in TRAIN set:")
for k, v in counter.items():
    print(f"{k}: {v}")
