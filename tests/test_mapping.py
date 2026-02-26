import json
from pathlib import Path

import yaml


DATASET_YAML = Path("config/datasets/dataset_a.yaml")
TRAIN_JSONL = Path("data/raw/uit_visd4sa/data/train.jsonl")


def parse_label(label_str: str):
    """
    Input:  "BATTERY#POSITIVE"
    Output: ("BATTERY", "POSITIVE")
    """
    if "#" not in label_str:
        return None, None
    a, s = label_str.split("#", 1)
    return a.strip(), s.strip()


def main():
    # 1) Load config
    cfg = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))

    text_field = cfg["text_field"]
    label_field = cfg["label_field"]
    aspect_mapping = cfg["aspect_mapping"]
    sentiment_mapping = cfg["sentiment_mapping"]
    ignored_aspects = set(cfg.get("ignored_aspects", []))
    target_aspects = cfg["target_aspects"]

    print("Loaded config OK:")
    print(" - target_aspects:", target_aspects)
    print(" - ignored_aspects:", sorted(list(ignored_aspects)))

    # 2) Read first sample
    first_line = TRAIN_JSONL.read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(first_line)

    text = obj[text_field]
    labels = obj[label_field]

    mapped = []
    skipped = []

    for start, end, raw_label in labels:
        raw_aspect, raw_sent = parse_label(raw_label)

        if raw_aspect in ignored_aspects:
            skipped.append((raw_aspect, raw_sent, "ignored_aspects"))
            continue

        if raw_aspect not in aspect_mapping:
            skipped.append((raw_aspect, raw_sent, "no_mapping"))
            continue

        if raw_sent not in sentiment_mapping:
            skipped.append((raw_aspect, raw_sent, "no_sentiment_mapping"))
            continue

        mapped.append(
            {
                "span": [start, end],
                "raw": raw_label,
                "aspect": aspect_mapping[raw_aspect],
                "sentiment": sentiment_mapping[raw_sent],
                "term": text[start:end],
            }
        )

    print("\n=== TEXT ===")
    print(text)

    print("\n=== MAPPED LABELS (kept) ===")
    for m in mapped:
        print(m)

    print("\n=== SKIPPED (debug) ===")
    for s in skipped:
        print(s)

    print("\nDONE.")


if __name__ == "__main__":
    main()
