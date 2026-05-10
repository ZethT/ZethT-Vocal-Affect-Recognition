"""
Model loading and inference for vocal affect classification.

The pipeline (StandardScaler + LogisticRegression) is loaded once at module
import time so the file is read from disk only on cold start, not per request.
"""

import pathlib
import joblib
import pandas as pd

from features import FEATURE_NAMES

_MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "model" / "vocal_model_lr.joblib"

pipeline = joblib.load(_MODEL_PATH)


def predict(features: dict) -> dict:
    """
    Run inference on a feature dict returned by extract_features().

    Returns:
        {
            "label": "high_energy" | "low_energy",
            "confidence": float,
            "probabilities": {"high_energy": float, "low_energy": float}
        }
    """
    X = pd.DataFrame([features], columns=FEATURE_NAMES)

    label = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    classes = pipeline.classes_

    prob_map = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence = round(float(max(proba)), 4)

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_map,
    }
