"""
Acoustic feature extraction for vocal affect classification.

All logic mirrors the training notebook (CS_163_Project.ipynb) exactly so that
inference features are computed the same way as training features.
"""

import numpy as np
import librosa
import parselmouth


FEATURE_NAMES = [
    "f0",
    "spectral_centroid",
    "spectral_rolloff",
    "hnr",
    "mfcc_0", "mfcc_1", "mfcc_2",  "mfcc_3",
    "mfcc_4", "mfcc_5", "mfcc_6",  "mfcc_7",
    "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11",
    "mfcc_12",
]

# Validation thresholds — mirror notebook data-cleaning rules
F0_MIN = 50.0    # Hz
F0_MAX = 500.0   # Hz
HNR_MIN = -20.0  # dB


def compute_f0_yin(y: np.ndarray, sr: int, fmin: float = F0_MIN, fmax: float = F0_MAX) -> float:
    """Return mean fundamental frequency (Hz) over voiced frames, or NaN if none found."""
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return float(np.mean(f0)) if len(f0) else float("nan")


def compute_hnr_praat(file_path: str, min_pitch: float = F0_MIN) -> float:
    """Return mean Harmonic-to-Noise Ratio (dB) via Praat cross-correlation, or NaN if none found."""
    snd = parselmouth.Sound(file_path)
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=min_pitch)
    vals = harm.values[0]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def extract_features(file_path: str) -> dict:
    """
    Extract a 17-dimensional acoustic feature vector from a .wav file.

    Returns a dict with keys matching FEATURE_NAMES.
    Raises ValueError for invalid or out-of-range feature values.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    f0 = compute_f0_yin(y, sr)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    hnr = compute_hnr_praat(file_path)
    mfcc_means = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # --- Validation (mirrors notebook cleaning logic) ---
    if not np.isfinite(f0) or not (F0_MIN <= f0 <= F0_MAX):
        raise ValueError(
            f"f0 out of valid range [{F0_MIN}–{F0_MAX} Hz]: got {f0:.2f}. "
            "Upload a clear singing sample with a detectable pitch."
        )
    if not np.isfinite(centroid) or centroid <= 0:
        raise ValueError(
            f"Spectral centroid is invalid ({centroid}). "
            "The audio may be silent or corrupted."
        )
    if not np.isfinite(rolloff) or rolloff <= 0:
        raise ValueError(
            f"Spectral rolloff is invalid ({rolloff}). "
            "The audio may be silent or corrupted."
        )
    if not np.isfinite(hnr) or hnr < HNR_MIN:
        raise ValueError(
            f"HNR below threshold ({hnr:.2f} dB < {HNR_MIN} dB). "
            "The audio may be too noisy or contain no harmonic content."
        )

    features = {
        "f0": f0,
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "hnr": hnr,
    }
    for i, val in enumerate(mfcc_means):
        features[f"mfcc_{i}"] = float(val)

    return features
