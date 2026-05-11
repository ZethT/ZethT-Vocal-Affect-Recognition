"""
Firestore client for persisting prediction logs.

Writes one document per prediction to the `predictions` collection.
Designed to fail soft: if credentials or the database are unavailable,
the caller logs a warning and the prediction response still succeeds.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore

logger = logging.getLogger(__name__)

_COLLECTION = "predictions"
_client: Optional[firestore.Client] = None
_init_attempted = False


def _get_client() -> Optional[firestore.Client]:
    """Lazily initialize the Firestore client on first use.

    Doing this at import time can stall Cloud Run startup probes when
    credentials or the metadata server are slow to respond.
    """
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True
    try:
        _client = firestore.Client()
    except Exception as e:
        logger.warning("Firestore client could not be initialized: %s", e)
        _client = None
    return _client


def save_prediction(data: dict) -> Optional[str]:
    """
    Persist a prediction record to Firestore.

    Expected keys in `data`:
        label, confidence, probabilities,
        f0, spectral_centroid, spectral_rolloff, hnr

    Returns the new document ID, or None if the write failed.
    """
    client = _get_client()
    if client is None:
        logger.warning("Firestore client unavailable; skipping save_prediction.")
        return None

    doc = {
        "timestamp": datetime.now(timezone.utc),
        "label": data.get("label"),
        "confidence": data.get("confidence"),
        "probabilities": data.get("probabilities"),
        "f0": data.get("f0"),
        "spectral_centroid": data.get("spectral_centroid"),
        "spectral_rolloff": data.get("spectral_rolloff"),
        "hnr": data.get("hnr"),
    }

    _, ref = client.collection(_COLLECTION).add(doc)
    return ref.id
