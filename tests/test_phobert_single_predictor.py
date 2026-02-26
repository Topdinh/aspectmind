from aspectmind.inference.phobert_single_predictor import PhoBERTSinglePredictor

p = PhoBERTSinglePredictor(
    ckpt_path="runs/phobert_single_2026-02-15_16-15-19/best_model.pt",
    threshold=0.5,
)

text = "Pin trâu, camera đẹp nhưng giá hơi cao."
print("TEXT:", text)
print("PRED:", p.predict(text))
print("PROBA:", p.predict_proba(text))
