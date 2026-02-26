# scripts/compare_models.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


@dataclass
class ModelResult:
    name: str
    macro_f1: float
    micro_f1: float
    sent_acc: Optional[float]  # only for multitask
    per_aspect_f1: Dict[str, float]


def _parse_float_after_colon(line: str) -> Optional[float]:
    # ex: "macro_f1 : 0.8150"
    if ":" not in line:
        return None
    try:
        return float(line.split(":", 1)[1].strip())
    except Exception:
        return None


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def parse_eval_output(txt: str, name: str) -> ModelResult:
    """
    Parse console output of:
      - scripts/eval_baseline.py
      - scripts/eval_phobert_single.py
      - scripts/eval_phobert_multitask.py
    We assume the output contains:
      macro_f1 : ...
      micro_f1 : ...
      per-aspect F1:
        - battery : ...
        ...
    Multi-task also contains:
      sent_acc : ... (n=...)
    """
    macro_f1 = None
    micro_f1 = None
    sent_acc = None
    per_aspect: Dict[str, float] = {}

    lines = [ln.rstrip() for ln in txt.splitlines()]

    # quick scan
    for ln in lines:
        s = ln.strip()

        if s.startswith("macro_f1"):
            macro_f1 = _parse_float_after_colon(s)
        elif s.startswith("micro_f1"):
            micro_f1 = _parse_float_after_colon(s)
        elif s.startswith("sent_acc"):
            # ex: "sent_acc : 0.8766 (n=3897)"
            if ":" in s:
                try:
                    sent_acc = float(s.split(":", 1)[1].split()[0].strip())
                except Exception:
                    sent_acc = None

        # per-aspect lines: "- battery : 0.9654"
        if s.startswith("- "):
            # remove "- "
            rest = s[2:].strip()
            if ":" in rest:
                a, v = rest.split(":", 1)
                a = a.strip()
                if a in ASPECTS:
                    try:
                        per_aspect[a] = float(v.strip())
                    except Exception:
                        pass

    if macro_f1 is None or micro_f1 is None:
        raise ValueError(
            f"Cannot parse macro/micro F1 from {name}. "
            f"Make sure you pass the correct log file content."
        )

    # ensure all aspects exist (default 0 if missing)
    for a in ASPECTS:
        per_aspect.setdefault(a, 0.0)

    return ModelResult(
        name=name,
        macro_f1=float(macro_f1),
        micro_f1=float(micro_f1),
        sent_acc=None if sent_acc is None else float(sent_acc),
        per_aspect_f1=per_aspect,
    )


def print_table(results: List[ModelResult]) -> None:
    # simple console table
    headers = ["Model", "macro_f1", "micro_f1", "sent_acc"]
    rows = []
    for r in results:
        rows.append(
            [
                r.name,
                f"{r.macro_f1:.4f}",
                f"{r.micro_f1:.4f}",
                "-" if r.sent_acc is None else f"{r.sent_acc:.4f}",
            ]
        )

    # compute widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print("\n=== OVERALL METRICS (TEST) ===")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))

    print("\n=== PER-ASPECT F1 (TEST) ===")
    # per-aspect table
    headers2 = ["Aspect"] + [r.name for r in results]
    widths2 = [len(h) for h in headers2]

    per_rows = []
    for a in ASPECTS:
        row = [a] + [f"{r.per_aspect_f1.get(a, 0.0):.4f}" for r in results]
        per_rows.append(row)
        for i, cell in enumerate(row):
            widths2[i] = max(widths2[i], len(cell))

    def fmt_row2(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths2[i]) for i, cell in enumerate(row))

    print(fmt_row2(headers2))
    print("-+-".join("-" * w for w in widths2))
    for row in per_rows:
        print(fmt_row2(row))


def plot_overall(results: List[ModelResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results]
    macro = [r.macro_f1 for r in results]
    micro = [r.micro_f1 for r in results]

    # macro_f1
    plt.figure()
    plt.bar(names, macro)
    plt.ylabel("macro_f1")
    plt.title("Macro-F1 (TEST)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "macro_f1_test.png", dpi=200)
    plt.close()

    # micro_f1
    plt.figure()
    plt.bar(names, micro)
    plt.ylabel("micro_f1")
    plt.title("Micro-F1 (TEST)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "micro_f1_test.png", dpi=200)
    plt.close()


def plot_per_aspect(results: List[ModelResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]

    # one chart per aspect (clean for report)
    for a in ASPECTS:
        vals = [r.per_aspect_f1.get(a, 0.0) for r in results]
        plt.figure()
        plt.bar(names, vals)
        plt.ylim(0.0, 1.0)
        plt.ylabel("F1")
        plt.title(f"Per-aspect F1 (TEST): {a}")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"f1_{a}_test.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline_log",
        type=str,
        required=True,
        help="Path to a text file that contains output of scripts/eval_baseline.py",
    )
    ap.add_argument(
        "--single_log",
        type=str,
        required=True,
        help="Path to a text file that contains output of scripts/eval_phobert_single.py",
    )
    ap.add_argument(
        "--multitask_log",
        type=str,
        required=True,
        help="Path to a text file that contains output of scripts/eval_phobert_multitask.py",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="runs/compare_models",
        help="Where to save plots",
    )
    args = ap.parse_args()

    baseline_txt = _read_text(Path(args.baseline_log))
    single_txt = _read_text(Path(args.single_log))
    multitask_txt = _read_text(Path(args.multitask_log))

    results = [
        parse_eval_output(baseline_txt, "Baseline"),
        parse_eval_output(single_txt, "PhoBERT Single"),
        parse_eval_output(multitask_txt, "PhoBERT Multi"),
    ]

    print_table(results)

    out_dir = Path(args.out_dir)
    plot_overall(results, out_dir)
    plot_per_aspect(results, out_dir)

    print(f"\n✅ Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
