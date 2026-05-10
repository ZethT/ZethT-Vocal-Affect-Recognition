"""
Sanity-check script for vocal_model_lr.joblib.

Run from the repo root:
    python backend/tests/verify_model.py

Confirms the model file loads and produces a valid prediction on a dummy
17-feature vector without error. Keep this script permanently — it is the
fastest way to confirm the model file is intact after any changes.
"""

import sys
import pathlib
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "model" / "vocal_model_lr.joblib"

FEATURE_NAMES = [
    "f0",
    "spectral_centroid",
    "spectral_rolloff",
    "hnr",
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3",
    "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7",
    "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11",
    "mfcc_12",
]

# Dummy values that resemble a typical belt (high-energy) recording
DUMMY_HIGH_ENERGY = [
    220.0,    # f0 (Hz) — mid-range pitch
    3200.0,   # spectral_centroid (Hz)
    6500.0,   # spectral_rolloff (Hz)
    12.5,     # hnr (dB) — relatively harmonic
    -200.0, 80.0, -30.0, 20.0,   # mfcc_0–3
    -10.0, 5.0, -5.0, 3.0,       # mfcc_4–7
    -2.0, 1.0, -1.0, 0.5, -0.5,  # mfcc_8–12
]


def main():
    try:
        import joblib
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}. Run: pip install joblib pandas")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    assert len(DUMMY_HIGH_ENERGY) == len(FEATURE_NAMES), (
        f"Feature count mismatch: expected {len(FEATURE_NAMES)}, got {len(DUMMY_HIGH_ENERGY)}"
    )

    # Use a DataFrame with named columns to match how the model was trained
    X = pd.DataFrame([DUMMY_HIGH_ENERGY], columns=FEATURE_NAMES)

    label = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    classes = pipeline.classes_

    print(f"\nPrediction:    {label}")
    print(f"Confidence:    {max(proba):.1%}")
    print("Probabilities:")
    for cls, p in zip(classes, proba):
        print(f"  {cls:<14} {p:.4f}")
    print("\nModel check passed.")


if __name__ == "__main__":
    main()
