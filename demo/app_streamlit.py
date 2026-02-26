import json
import math
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


# ----------------------------
# Temperature scaling helpers
# ----------------------------
@st.cache_data
def _load_temperature_value(path: str) -> float:
    """
    Load temperature from temperature_*.json.
    Expected format:
      {"temperature": 0.867545, ...}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "temperature" not in data:
        raise ValueError(f"Invalid temperature file: {path} (missing 'temperature')")
    t = float(data["temperature"])
    if t <= 0:
        raise ValueError(f"Temperature must be > 0, got {t}")
    return t


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    eps = 1e-7
    p = max(min(float(p), 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def apply_temperature_scaling(proba: Dict[str, float], temperature: float) -> Dict[str, float]:
    """
    Apply temperature scaling in probability space:
      p -> logit(p) -> (logit/T) -> sigmoid
    """
    T = max(float(temperature), 1e-6)
    out: Dict[str, float] = {}
    for a, p in proba.items():
        z = _logit(float(p))
        out[a] = float(_sigmoid(z / T))
    return out


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

    st.title("AspectMind Demo")
    st.caption("Aspect Detection (6 aspects) • Switchable models: Baseline vs PhoBERT")

    # ---- Model selector ----
    model_choice = st.selectbox(
        "Chọn model",
        ["Baseline (TF-IDF + LR)", "PhoBERT Single-task", "PhoBERT Multi-task"],
    )

    # ---- Calibration controls (only for PhoBERT Single) ----
    # NOTE: keep outside cache so toggles don't force model reload
    phobert_single_ckpt = "runs/phobert_single_2026-02-15_16-15-19/best_model.pt"
    phobert_single_thr_path = "runs/phobert_single_2026-02-15_16-15-19/thresholds_dev.json"
    phobert_single_temp_path = "runs/phobert_single_2026-02-15_16-15-19/temperature_dev.json"

    use_calibrated = False
    use_temperature = False
    temperature_value: Optional[float] = None

    if model_choice == "PhoBERT Single-task":
        with st.expander("Calibration (PhoBERT Single-task)", expanded=True):
            use_calibrated = st.checkbox(
                "Use calibrated per-aspect thresholds (from dev)",
                value=True,
                help="Bật để dùng thresholds_dev.json (per-aspect). Tắt để dùng threshold global 0.5.",
            )

            tp = Path(phobert_single_thr_path)
            if use_calibrated:
                if tp.exists():
                    st.caption(f"✅ Using calibrated thresholds: `{phobert_single_thr_path}`")
                else:
                    st.warning(
                        f"Không tìm thấy file thresholds: `{phobert_single_thr_path}`. "
                        "Demo sẽ fallback về threshold=0.5."
                    )

            # NEW: temperature scaling toggle
            use_temperature = st.checkbox(
                "Use temperature scaling (from dev)",
                value=False,
                help="Bật để scale xác suất bằng T fit trên dev (cải thiện calibration/ECE).",
            )

            tpath = Path(phobert_single_temp_path)
            if use_temperature:
                if tpath.exists():
                    try:
                        temperature_value = _load_temperature_value(phobert_single_temp_path)
                        st.caption(f"✅ Using temperature: `T={temperature_value:.6f}` from `{phobert_single_temp_path}`")
                    except Exception as e:
                        st.error(f"Lỗi đọc temperature file: {e}")
                        use_temperature = False
                        temperature_value = None
                else:
                    st.warning(
                        f"Không tìm thấy file temperature: `{phobert_single_temp_path}`. "
                        "Temperature scaling sẽ bị tắt."
                    )
                    use_temperature = False
                    temperature_value = None

    # ---- Load predictors (cache) ----
    @st.cache_resource
    def get_baseline():
        return BaselinePredictor("runs/baseline")

    @st.cache_resource
    def get_phobert_single(use_thresholds_file: bool):
        # cache key depends on use_thresholds_file to avoid mixing objects
        thresholds_path = (
            phobert_single_thr_path
            if use_thresholds_file and Path(phobert_single_thr_path).exists()
            else None
        )
        return PhoBERTSinglePredictor(
            ckpt_path=phobert_single_ckpt,
            threshold=0.5,
            thresholds_path=thresholds_path,
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
    if model_choice == "Baseline (TF-IDF + LR)":
        predictor = get_baseline()
    elif model_choice == "PhoBERT Single-task":
        predictor = get_phobert_single(use_thresholds_file=use_calibrated)
    else:
        predictor = get_phobert_multi()

    samples = load_samples()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Input")
        default_text = samples[0] if samples else "Pin trâu, camera đẹp nhưng giá hơi cao."
        text = st.text_area("Nhập review tiếng Việt:", value=default_text, height=140)

        c1, c2, c3 = st.columns([1, 1, 2])
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
            st.write("Outputs:")
            st.markdown("- `PRED`: 0/1 cho mỗi aspect\n- `PROBA`: xác suất aspect xuất hiện")

        elif model_choice == "PhoBERT Single-task":
            st.code("phobert_single (runs/phobert_single_2026-02-15_16-15-19)", language="text")
            st.caption("vinai/phobert-base + Linear head (multi-label BCE)")
            st.write("Outputs:")
            st.markdown("- `PRED`: 0/1 cho mỗi aspect\n- `PROBA`: xác suất aspect xuất hiện")

            # show threshold mode
            if use_calibrated and Path(phobert_single_thr_path).exists():
                thr_msg = "Threshold mode: **Calibrated per-aspect (dev)**"
            else:
                thr_msg = "Threshold mode: **Global threshold = 0.5**"

            if use_temperature and temperature_value is not None:
                temp_msg = f"Temperature scaling: **ON** (T={temperature_value:.4f})"
            else:
                temp_msg = "Temperature scaling: **OFF**"

            st.info(thr_msg + "\n\n" + temp_msg)

        else:
            st.code("phobert_multitask (runs/phobert_multitask_2026-02-15_23-44-18)", language="text")
            st.caption("vinai/phobert-base + 2 heads (aspect multi-label + per-aspect sentiment)")
            st.write("Outputs:")
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

        # ----- Predict -----
        sent = None
        sent_proba = None

        if model_choice == "PhoBERT Multi-task":
            out = predictor.predict_with_sentiment(text)
            pred = out.pred_aspect
            proba = out.proba_aspect
            sent = out.sent
            sent_proba = out.sent_proba

        elif model_choice == "PhoBERT Single-task":
            # Get raw proba from model
            proba_raw = predictor.predict_proba(text)

            # Optionally apply temperature scaling to proba
            proba = proba_raw
            if use_temperature and temperature_value is not None:
                proba = apply_temperature_scaling(proba_raw, temperature_value)

            # IMPORTANT:
            # predictor.predict(text, use_calibrated=...) would recompute proba internally (raw),
            # so for temperature demo we threshold HERE using the displayed proba.
            per_thr = getattr(predictor, "per_aspect_thresholds", None)
            global_thr = float(getattr(predictor, "threshold", 0.5))

            pred = _threshold_predict(
                proba=proba,
                global_thr=global_thr,
                per_aspect_thr=per_thr if isinstance(per_thr, dict) else None,
                use_calibrated=use_calibrated,
            )

            # Optional debug compare
            with st.expander("Debug: proba raw vs after temperature", expanded=False):
                if use_temperature and temperature_value is not None:
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
            row = {
                "aspect": a,
                "pred": int(pred[a]),
                "proba": float(proba[a]),
            }
            if model_choice == "PhoBERT Multi-task":
                row["sent"] = sent[a]
                rows.append(row)
            else:
                rows.append(row)

        sort_cols = ["pred", "proba"]
        df = pd.DataFrame(rows).sort_values(sort_cols, ascending=[False, False]).reset_index(drop=True)

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
            # show bars
            for _, r in df.iterrows():
                label = f"{r['aspect']}  (pred={r['pred']})"
                st.progress(min(max(float(r["proba"]), 0.0), 1.0), text=label)

        st.markdown("### Raw text")
        st.write(text)


if __name__ == "__main__":
    main()