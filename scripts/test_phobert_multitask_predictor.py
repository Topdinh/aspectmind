from aspectmind.inference.phobert_multitask_predictor import PhoBERTMultiTaskPredictor

RUN_DIR = r"runs/phobert_multitask_2026-02-15_23-44-18"

p = PhoBERTMultiTaskPredictor(RUN_DIR)

text = "Pin trâu, camera đẹp nhưng giá hơi cao."
out = p.predict_with_sentiment(text)

print("TEXT:", text)
print("PRED_ASPECT:", out.pred_aspect)
print("PROBA_ASPECT:", out.proba_aspect)
print("SENT:", out.sent)
print("SENT_PROBA:", out.sent_proba)
