"""
FastAPI inference service for vocal affect classification.

Routes:
    GET  /health   — liveness check
    POST /predict  — accepts a .wav upload, returns prediction JSON
"""

import logging
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from features import extract_features
from model import predict
from firestore_client import save_prediction

logger = logging.getLogger(__name__)

app = FastAPI(title="Vocal Affect Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=422, detail="Only .wav files are supported.")

    # Write upload to a temp file so librosa and parselmouth can read it by path
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)

    result = predict(features)

    # Persist to Firestore — fail soft so a DB error never breaks the demo
    try:
        save_prediction({
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "f0": features["f0"],
            "spectral_centroid": features["spectral_centroid"],
            "spectral_rolloff": features["spectral_rolloff"],
            "hnr": features["hnr"],
        })
    except Exception as e:
        logger.warning("Firestore save_prediction failed: %s", e)

    return {
        **result,
        "features": {
            "f0":                round(features["f0"], 2),
            "spectral_centroid": round(features["spectral_centroid"], 2),
            "spectral_rolloff":  round(features["spectral_rolloff"], 2),
            "hnr":               round(features["hnr"], 2),
        },
    }
