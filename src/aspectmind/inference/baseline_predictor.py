from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import joblib


ASPECTS: List[str] = ["battery", "camera", "performance", "design", "price", "service"]


@dataclass
class BaselinePredictor:
    artifact_dir: Union[str, Path] = Path("runs/baseline")

    def __post_init__(self):
        self.artifact_dir = Path(self.artifact_dir)

        vec_path = self.artifact_dir / "tfidf_vectorizer.joblib"
        model_path = self.artifact_dir / "baseline_ovr_lr.joblib"

        if not vec_path.exists():
            raise FileNotFoundError(f"Missing vectorizer: {vec_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")

        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> Dict[str, int]:
        X = self.vectorizer.transform([text])
        y = self.model.predict(X)[0]
        return {a: int(v) for a, v in zip(ASPECTS, y)}

    def predict_proba(self, text: str) -> Dict[str, float]:
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        return {a: float(p) for a, p in zip(ASPECTS, proba)}
