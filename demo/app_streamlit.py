import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from aspectmind.inference.baseline_predictor import BaselinePredictor
from aspectmind.inference.phobert_single_predictor import PhoBERTSinglePredictor
from aspectmind.inference.phobert_multitask_predictor import PhoBERTMultiTaskPredictor

ASPECTS = ["battery", "camera", "performance", "design", "price", "service"]


def load_samples():
    p = Path("demo/sample_reviews.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _threshold_predict(
    proba: Dict[str, float],
    global_thr: float,
    per_aspect_thr: Optional[Dict[str, float]] = None,
    use_calibrated: bool = False,
) -> Dict[str, int]:
    pred: Dict[str, int] = {}
    for a in ASPECTS:
        thr = float(global_thr)
        if use_calibrated and isinstance(per_aspect_thr, dict):
            thr = float(per_aspect_thr.get(a, global_thr))
        pred[a] = int(float(proba[a]) >= thr)
    return pred


def main():
    st.set_page_config(page_title="AspectMind Demo", layout="wide")

    # Reduce noisy HF logs (best-effort). These env vars are safe.
    # (They only affect logging/telemetry; do not change model outputs.)
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    st.title("AspectMind Demo")
    st.caption("Aspect Detection (6 aspects) • Switchable models: Baseline vs PhoBERT")

    # ---- Model selector ----
    model_choice = st.selectbox(
        "Chọn model",
        ["Baseline (TF-IDF + LR)", "PhoBERT Single-task", "PhoBERT Multi-task"],
    )

    # ---- Paths for Single-task run ----
    phobert_single_run_dir = Path("runs/phobert_single_2026-02-15_16-15-19")
    phobert_single_ckpt = phobert_single_run_dir / "best_model.pt"
    phobert_single_thr_path = phobert_single_run_dir / "thresholds_dev.json"
    phobert_single_temp_path = phobert_single_run_dir / "temperature_dev.json"

    # ---- Calibration controls (only for PhoBERT Single) ----
    use_calibrated = False
    use_temperature = False

    if model_choice == "PhoBERT Single-task":
        with st.expander("Calibration (PhoBERT Single-task)", expanded=True):
            use_calibrated = st.checkbox(
                "Use calibrated per-aspect thresholds (from dev)",
                value=True,
                help="Bật để dùng thresholds_dev.json (per-aspect). Tắt để dùng threshold global 0.5.",
            )

            if use_calibrated:
                if phobert_single_thr_path.exists():
                    st.caption(f"✅ thresholds: `{phobert_single_thr_path.as_posix()}`")
                else:
                    st.warning(
                        f"Không tìm thấy file thresholds: `{phobert_single_thr_path.as_posix()}`. "
                        "Demo sẽ fallback về threshold=0.5."
                    )

            use_temperature = st.checkbox(
                "Use temperature scaling (from dev)",
                value=False,
                help="Bật để apply temperature scaling trên logits (cải thiện calibration/ECE).",
            )

            if use_temperature:
                if phobert_single_temp_path.exists():
                    st.caption(f"✅ temperature file: `{phobert_single_temp_path.as_posix()}`")
                else:
                    st.warning(
                        f"Không tìm thấy file temperature: `{phobert_single_temp_path.as_posix()}`. "
                        "Temperature scaling sẽ bị tắt."
                    )
                    use_temperature = False

    # ---- Load predictors (cache) ----
    @st.cache_resource
    def get_baseline():
        return BaselinePredictor("runs/baseline")

    @st.cache_resource
    def get_phobert_single():
        # Predictor mới sẽ auto-load thresholds_dev.json + temperature_dev.json từ run_dir
        return PhoBERTSinglePredictor(
            ckpt_path=phobert_single_ckpt,
            threshold=0.5,
            thresholds_path=None,      # auto
            temperature_path=None,     # auto
        )

    @st.cache_resource
    def get_phobert_multi():
        return PhoBERTMultiTaskPredictor(
            run_dir="runs/phobert_multitask_2026-02-15_23-44-18",
            ckpt="vinai/phobert-base",
            max_length=256,
            aspect_threshold=0.5,
        )

    # Resolve predictor
    predictor = None
    if model_choice == "Baseline (TF-IDF + LR)":
        predictor = get_baseline()
    elif model_choice == "PhoBERT Single-task":
        predictor = get_phobert_single()
    else:
        predictor = get_phobert_multi()

    samples = load_samples()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Input")
        default_text = samples[0] if samples else "Pin trâu, camera đẹp nhưng giá hơi cao."
        text = st.text_area("Nhập review tiếng Việt:", value=default_text, height=140)

        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            run_btn = st.button("Predict", type="primary")
        with c2:
            if samples:
                pick = st.selectbox(
                    "Hoặc chọn sample:",
                    options=list(range(len(samples))),
                    format_func=lambda i: samples[i][:60] + ("..." if len(samples[i]) > 60 else ""),
                )
                if st.button("Load sample"):
                    st.session_state["text"] = samples[pick]
            else:
                st.write("")

        # persist text if needed
        if "text" in st.session_state:
            text = st.session_state["text"]

    with col_right:
        st.subheader("Model")

        if model_choice == "Baseline (TF-IDF + LR)":
            st.code("baseline (runs/baseline)", language="text")
            st.caption("TF-IDF char n-grams + One-vs-Rest Logistic Regression")
            st.markdown("- `PRED`: 0/1 cho mỗi aspect\n- `PROBA`: xác suất aspect xuất hiện")

        elif model_choice == "PhoBERT Single-task":
            st.code(f"phobert_single ({phobert_single_run_dir.as_posix()})", language="text")
            st.caption("vinai/phobert-base + Linear head (multi-label BCE)")
            st.markdown("- `PRED`: 0/1 cho mỗi aspect\n- `PROBA`: xác suất aspect xuất hiện")

            # status panel (robust)
            status = {}
            if hasattr(predictor, "status") and callable(getattr(predictor, "status")):
                try:
                    status = predictor.status()
                except Exception:
                    status = {}

            # threshold mode (what user toggled + file availability)
            has_thr = bool(status.get("has_per_aspect_thresholds", False))
            if use_calibrated and has_thr:
                thr_msg = "Threshold mode: **Calibrated per-aspect (dev)**"
            else:
                thr_msg = "Threshold mode: **Global threshold = 0.5**"

            # temperature mode (what user toggled + file availability)
            has_temp = status.get("temperature", None) is not None
            if use_temperature and has_temp:
                temp_msg = f"Temperature scaling: **ON** (T={float(status['temperature']):.4f})"
            elif use_temperature and not has_temp:
                temp_msg = "Temperature scaling: **ON (requested)** but **temperature file missing → OFF (fallback)**"
            else:
                temp_msg = "Temperature scaling: **OFF**"

            st.info(thr_msg + "\n\n" + temp_msg)

            # show compact status
            with st.expander("Runtime status", expanded=False):
                if status:
                    st.json(status)
                else:
                    st.write("No status available.")

        else:
            st.code("phobert_multitask (runs/phobert_multitask_2026-02-15_23-44-18)", language="text")
            st.caption("vinai/phobert-base + 2 heads (aspect multi-label + per-aspect sentiment)")
            st.markdown(
                "- `PRED`: 0/1 cho mỗi aspect\n"
                "- `PROBA`: xác suất aspect xuất hiện\n"
                "- `SENT`: sentiment cho aspect (pos/neg/neu, None nếu aspect không xuất hiện)\n"
                "- `SENT_PROBA`: xác suất sentiment (pos/neg/neu, None nếu aspect không xuất hiện)"
            )

    st.divider()
    st.subheader("Output")

    if run_btn:
        if not text.strip():
            st.warning("Bạn chưa nhập review.")
            st.stop()

        sent = None
        sent_proba = None

        # ----- Predict -----
        if model_choice == "PhoBERT Multi-task":
            out = predictor.predict_with_sentiment(text)
            pred = out.pred_aspect
            proba = out.proba_aspect
            sent = out.sent
            sent_proba = out.sent_proba

        elif model_choice == "PhoBERT Single-task":
            # IMPORTANT: temperature scaling MUST be applied on logits before sigmoid
            # Predictor supports: predict_proba(text, use_temperature=...)
            proba_raw = predictor.predict_proba(text, use_temperature=False)

            # If user toggles temp scaling, ask predictor for calibrated probs
            proba = proba_raw
            if use_temperature:
                # predictor will fallback internally if temperature missing
                proba = predictor.predict_proba(text, use_temperature=True)

            # IMPORTANT: To reflect the displayed proba, threshold HERE
            per_thr = getattr(predictor, "per_aspect_thresholds", None)
            global_thr = float(getattr(predictor, "threshold", 0.5))

            pred = _threshold_predict(
                proba=proba,
                global_thr=global_thr,
                per_aspect_thr=per_thr if isinstance(per_thr, dict) else None,
                use_calibrated=use_calibrated,
            )

            # Debug compare raw vs after temperature
            with st.expander("Debug: proba raw vs after temperature", expanded=False):
                if use_temperature:
                    dbg = []
                    for a in ASPECTS:
                        dbg.append(
                            {"aspect": a, "raw_proba": float(proba_raw[a]), "after_temp": float(proba[a])}
                        )
                    st.dataframe(pd.DataFrame(dbg), use_container_width=True)
                else:
                    st.write("Temperature scaling is OFF.")

        else:
            pred = predictor.predict(text)
            proba = predictor.predict_proba(text)

        # ----- Build table -----
        rows = []
        for a in ASPECTS:
            row = {"aspect": a, "pred": int(pred[a]), "proba": float(proba[a])}
            if model_choice == "PhoBERT Multi-task":
                row["sent"] = sent[a]
            rows.append(row)

        df = (
            pd.DataFrame(rows)
            .sort_values(["pred", "proba"], ascending=[False, False])
            .reset_index(drop=True)
        )

        cA, cB = st.columns([1, 1])
        with cA:
            st.markdown("### Prediction table")
            st.dataframe(df, use_container_width=True)

            if model_choice == "PhoBERT Multi-task":
                st.markdown("### Sentiment details (per predicted aspect)")
                for a in ASPECTS:
                    if sent[a] is None:
                        continue
                    st.write(f"**{a}** → `{sent[a]}`")
                    sp = sent_proba[a]
                    if isinstance(sp, dict):
                        st.json(sp)

        with cB:
            st.markdown("### Confidence")
            for _, r in df.iterrows():
                label = f"{r['aspect']}  (pred={r['pred']})"
                st.progress(min(max(float(r["proba"]), 0.0), 1.0), text=label)

        st.markdown("### Raw text")
        st.write(text)


if __name__ == "__main__":
    main()