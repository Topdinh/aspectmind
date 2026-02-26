from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch.utils.data import DataLoader

# ✅ theo project bạn
from aspectmind.data.transformer_dataset import build_phobert_datasets, collate_batch

# Tuner
from aspectmind.eval.threshold_tuning import ASPECTS, tune_thresholds_from_probs


# ----------------------------
# Checkpoint loader (robust)
# ----------------------------
def load_single_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    # support formats:
    # - raw state_dict
    # - {"model_state_dict": ...}
    # - {"state_dict": ...}
    if isinstance(state, dict):
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            state = state["model_state_dict"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

    # strip common prefix "model." if exists
    if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
        state = {k[len("model.") :]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[tune_threshold_single] Missing keys: {missing}")
    if unexpected:
        print(f"[tune_threshold_single] Unexpected keys: {unexpected}")


# ----------------------------
# Model resolver (robust)
# ----------------------------
def _resolve_single_model_class(preferred: Optional[str] = None):
    """
    Find the single-task model class inside aspectmind.models.phobert_single
    without assuming exact export name.
    """
    mod = importlib.import_module("aspectmind.models.phobert_single")

    if preferred:
        if hasattr(mod, preferred):
            cls = getattr(mod, preferred)
            if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
                return cls
        raise ImportError(f"Model class '{preferred}' not found in aspectmind.models.phobert_single")

    common = [
        "PhoBERTSingle",
        "PhoBERTSingleModel",
        "PhoBERTSingleTask",
        "PhoBERTSingleClassifier",
        "PhoBERTAspectClassifier",
        "PhoBERTForAspects",
        "PhoBERTForAspect",
    ]
    for name in common:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
                return cls

    candidates = []
    for _, obj in vars(mod).items():
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
            if obj.__module__ == mod.__name__:
                candidates.append(obj)

    if not candidates:
        raise ImportError("No torch.nn.Module subclass found in aspectmind.models.phobert_single")

    candidates.sort(key=lambda c: (("PhoBERT" not in c.__name__), c.__name__))
    return candidates[0]


def _instantiate_single_model(model_cls: type, ckpt: str, num_aspects: int) -> torch.nn.Module:
    tries = [
        dict(ckpt=ckpt, num_aspects=num_aspects),
        dict(model_name=ckpt, num_aspects=num_aspects),
        dict(ckpt=ckpt),
        dict(model_name=ckpt),
        {},
    ]
    last_err: Optional[Exception] = None
    for kwargs in tries:
        try:
            return model_cls(**kwargs)
        except TypeError as e:
            last_err = e
            continue
    raise TypeError(f"Cannot instantiate {model_cls.__name__}. Last error: {last_err}")


# ----------------------------
# Output extractor (robust)
# ----------------------------
def _extract_logits(out: Any) -> torch.Tensor:
    """
    Trả về logits dạng Tensor.
    Hỗ trợ:
      - Tensor
      - (logits, ...)
      - {"logits": ..., ...}
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if isinstance(out, dict):
        for k in ["logits", "aspect_logits", "pred_logits"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
    raise TypeError(f"Unsupported model output type: {type(out)}")


def _ensure_logits_shape(logits: torch.Tensor, num_aspects: int) -> torch.Tensor:
    """
    Chuẩn hoá logits về shape (B, num_aspects) nếu có thể.
    Nếu logits có shape (B, num_aspects, 1) hoặc (B, 1, num_aspects) -> squeeze/reshape hợp lý.
    """
    if logits.dim() == 2 and logits.size(-1) == num_aspects:
        return logits
    if logits.dim() == 3:
        # (B, A, 1) -> (B, A)
        if logits.size(1) == num_aspects and logits.size(2) == 1:
            return logits.squeeze(-1)
        # (B, 1, A) -> (B, A)
        if logits.size(2) == num_aspects and logits.size(1) == 1:
            return logits.squeeze(1)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)} (expected (B,{num_aspects}))")


@torch.no_grad()
def collect_probs_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_aspects: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      y_true: (N,6) int64 in {0,1}
      y_prob: (N,6) float32 (sigmoid probs)
    """
    all_true = []
    all_prob = []

    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _extract_logits(out)
        logits = _ensure_logits_shape(logits, num_aspects=num_aspects)

        probs = torch.sigmoid(logits).detach().cpu()

        y = batch["labels"].detach().cpu()
        # Nếu label có -1 (ignore) => chuyển về 0 để tune threshold không bị "rụng" F1
        y = y.long()
        y = torch.clamp(y, min=0, max=1)

        all_true.append(y)
        all_prob.append(probs)

    y_true = torch.cat(all_true, dim=0)
    y_prob = torch.cat(all_prob, dim=0)
    return y_true, y_prob


# ----------------------------
# ✅ FIX: patch load_split signature drift (PATCH ĐÚNG REFERENCE + KHÔNG TRÙNG KWARGS)
# ----------------------------
def _default_sentiment_mapping() -> Dict[str, int]:
    return {
        "pos": 0,
        "positive": 0,
        "POS": 0,
        "Pos": 0,
        "neg": 1,
        "negative": 1,
        "NEG": 1,
        "Neg": 1,
        "neu": 2,
        "neutral": 2,
        "NEU": 2,
        "Neu": 2,
    }


def _make_load_split_compat_wrapper(original_func, target_aspects: List[str]):
    """
    Wrapper an toàn:
    - thử gọi original
    - nếu TypeError do thiếu tham số (target_aspects / sentiment_mapping / ignored_aspects)
      -> dùng signature.bind_partial để biết tham số nào đã được bind (kể cả positional)
      -> chỉ inject những param còn thiếu (không đụng file_path hay các arg khác)
    """
    if getattr(original_func, "_aspectmind_patched", False):
        return original_func

    sig = inspect.signature(original_func)
    sentiment_mapping = _default_sentiment_mapping()
    ignored_aspects: List[str] = []

    def wrapper(*args, **kwargs):
        # 1) Try original first
        try:
            return original_func(*args, **kwargs)
        except TypeError as e1:
            # 2) Attempt to inject only known-missing params
            try:
                bound = sig.bind_partial(*args, **kwargs)  # respects positional binding
            except TypeError:
                # nếu bind_partial fail thì trả lại lỗi gốc cho rõ ràng
                raise e1

            # Inject ONLY if missing
            for name, p in sig.parameters.items():
                lname = name.lower()
                if name in bound.arguments:
                    continue  # already provided (positional or kw)

                if lname == "target_aspects" or ("target" in lname and "aspect" in lname):
                    bound.arguments[name] = target_aspects
                elif "sentiment" in lname:
                    bound.arguments[name] = sentiment_mapping
                elif "ignored" in lname and "aspect" in lname:
                    bound.arguments[name] = ignored_aspects
                else:
                    # do NOT fill other params (like file_path) to avoid duplicates/incorrect values
                    continue

            # Try again with injected args
            try:
                return original_func(*bound.args, **bound.kwargs)
            except TypeError:
                # if still fails, re-raise original error message (more informative)
                raise

    setattr(wrapper, "_aspectmind_patched", True)
    return wrapper


def _patch_load_split_everywhere(target_aspects: List[str]) -> None:
    """
    Patch cả 2 nơi:
      - aspectmind.data.loader.load_split
      - aspectmind.data.transformer_dataset.load_split
    để tránh trường hợp transformer_dataset đã bind load_split từ trước.
    """
    patched = []

    # 1) loader
    try:
        loader_mod = importlib.import_module("aspectmind.data.loader")
        if hasattr(loader_mod, "load_split") and callable(loader_mod.load_split):
            loader_mod.load_split = _make_load_split_compat_wrapper(loader_mod.load_split, target_aspects)
            patched.append("aspectmind.data.loader")
    except Exception:
        pass

    # 2) transformer_dataset reference (QUAN TRỌNG)
    try:
        td_mod = importlib.import_module("aspectmind.data.transformer_dataset")
        if hasattr(td_mod, "load_split") and callable(td_mod.load_split):
            td_mod.load_split = _make_load_split_compat_wrapper(td_mod.load_split, target_aspects)
            patched.append("aspectmind.data.transformer_dataset")
    except Exception:
        pass

    if patched:
        print(f"[tune_threshold_single] Patched load_split in: {', '.join(patched)}")
    else:
        print("[tune_threshold_single] Could not patch load_split (not found).")


# ----------------------------
# Sanity check
# ----------------------------
def _sanity_report(y_true: torch.Tensor, y_prob: torch.Tensor) -> None:
    print("\n[SANITY] y_true shape:", tuple(y_true.shape), "dtype:", y_true.dtype)
    print("[SANITY] y_prob shape:", tuple(y_prob.shape), "dtype:", y_prob.dtype)

    pos_per_aspect = y_true.sum(dim=0)
    print("[SANITY] positives per aspect:")
    for i, a in enumerate(ASPECTS):
        print(f"  - {a:12s}: {int(pos_per_aspect[i].item())}")

    print(
        "[SANITY] prob min/max/mean:",
        float(y_prob.min().item()),
        float(y_prob.max().item()),
        float(y_prob.mean().item()),
    )

    for thr in [0.05, 0.10, 0.20, 0.50]:
        pred = (y_prob >= thr).long()
        print(f"[SANITY] thr={thr:.2f} -> predicted positives:", int(pred.sum().item()))


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="runs/.../best_model.pt")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="vinai/phobert-base")
    ap.add_argument("--model_class", type=str, default="", help="Optional: force model class name inside phobert_single.py")
    ap.add_argument("--out_dir", type=str, default="", help="where to save thresholds.json (default: same dir as ckpt)")
    ap.add_argument("--no_sanity", action="store_true", help="Disable sanity prints")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ✅ Patch compat BEFORE building datasets
    _patch_load_split_everywhere(target_aspects=list(ASPECTS))

    # Bundle
    bundle = build_phobert_datasets(
        model_name=args.ckpt,
        max_length=args.max_length,
        use_fast=False,
    )

    if args.split == "train":
        ds = bundle.train_dataset
    elif args.split == "dev":
        ds = bundle.dev_dataset
    else:
        ds = bundle.test_dataset

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(bundle.tokenizer, b),
        num_workers=0,
    )

    preferred = args.model_class.strip() or None
    model_cls = _resolve_single_model_class(preferred=preferred)
    print(f"[tune_threshold_single] Using model class: {model_cls.__name__}")

    model = _instantiate_single_model(model_cls, ckpt=args.ckpt, num_aspects=len(ASPECTS)).to(device)
    load_single_checkpoint(model, ckpt_path)

    y_true, y_prob = collect_probs_and_labels(model, loader, device, num_aspects=len(ASPECTS))

    if not args.no_sanity:
        _sanity_report(y_true, y_prob)

    best_global, best_per = tune_thresholds_from_probs(y_true=y_true, y_prob=y_prob)

    print("\n=== BEST GLOBAL THRESHOLD (maximize macro-F1 on SPLIT) ===")
    print(f"thr={best_global.global_thr}")
    print(f"macro_f1={best_global.macro_f1:.4f} | micro_f1={best_global.micro_f1:.4f}")
    for a in ASPECTS:
        print(f"  - {a:12s}: {best_global.per_aspect_f1.get(a, 0.0):.4f}")

    print("\n=== BEST PER-ASPECT THRESHOLDS (maximize per-aspect F1) ===")
    print(f"macro_f1={best_per.macro_f1:.4f} | micro_f1={best_per.micro_f1:.4f}")
    for a in ASPECTS:
        print(f"  - {a:12s}: thr={best_per.per_aspect_thr[a]:.2f} | f1={best_per.per_aspect_f1.get(a, 0.0):.4f}")

    out_dir = Path(args.out_dir) if args.out_dir.strip() else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"thresholds_{args.split}.json"

    payload: Dict[str, object] = {
        "split": args.split,
        "ckpt_path": str(ckpt_path),
        "model_class": model_cls.__name__,
        "best_global": {
            "thr": best_global.global_thr,
            "macro_f1": best_global.macro_f1,
            "micro_f1": best_global.micro_f1,
            "per_aspect_f1": best_global.per_aspect_f1,
        },
        "best_per_aspect": {
            "thr": best_per.per_aspect_thr,
            "macro_f1": best_per.macro_f1,
            "micro_f1": best_per.micro_f1,
            "per_aspect_f1": best_per.per_aspect_f1,
        },
        "aspects": ASPECTS,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Saved: {out_json}")


if __name__ == "__main__":
    main()