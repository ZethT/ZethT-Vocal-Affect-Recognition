"""
FastAPI inference service for vocal affect classification.

Routes:
    GET  /health   — liveness check
    POST /predict  — accepts a .wav upload, returns prediction JSON
"""

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from features import extract_features
from model import predict

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

    return {
        **result,
        "features": {
            "f0":                round(features["f0"], 2),
            "spectral_centroid": round(features["spectral_centroid"], 2),
            "spectral_rolloff":  round(features["spectral_rolloff"], 2),
            "hnr":               round(features["hnr"], 2),
        },
    }
