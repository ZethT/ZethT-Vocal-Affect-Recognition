# Vocal Affect Recognition: Timbral Mood Classification

**CS 163 — Data Science Senior Project | SJSU | Group 16**  
Tanzil Ahmed · James Kim · Zeth Tang

A full-stack machine learning project that classifies vocal energy level (high vs. low) from singing audio using acoustic feature extraction and supervised classification. The deployed model achieves **91.4% accuracy** (Logistic Regression + MFCCs) on the VocalSet dataset.

**Live Website:** https://vocalaffect.uc.r.appspot.com

---

## What Is This Repo?

This repository contains the end-to-end pipeline for the Vocal Affect Recognition project: from raw audio feature extraction in a Jupyter notebook, to a trained scikit-learn model, to a containerized FastAPI inference service deployed on Google Cloud Run, to a static frontend hosted on Google App Engine.

The core question: can a lightweight acoustic classifier distinguish between *belt* (high-energy, powerful) and *breathy* (low-energy, airy) singing techniques using only a small set of acoustic measurements? The answer is yes, with 91.4% accuracy using 17 features (4 spectral + 13 MFCCs).

---

## Directory Structure

```
.
├── backend/                    # FastAPI inference service (deployed to Cloud Run)
│   ├── main.py                 # API entry point: /health and /predict endpoints
│   ├── features.py             # Audio feature extraction (librosa + parselmouth)
│   ├── model.py                # Model loading and inference logic
│   ├── firestore_client.py     # Firestore prediction logging (fail-soft)
│   ├── requirements.txt        # Python dependencies for the backend
│   ├── Dockerfile              # Container definition for Cloud Run
│   └── tests/
│       └── verify_model.py     # Sanity-check script for the loaded model
│
├── frontend/                   # Static website (deployed to Google App Engine)
│   ├── index.html              # Homepage: project objective and results overview
│   ├── eda.html                # Exploratory Data Analysis page
│   ├── methods.html            # Pipeline and modeling methodology page
│   ├── findings.html           # Major findings, charts, and conclusions
│   ├── demo.html               # Live inference demo (calls Cloud Run API)
│   ├── style.css               # Shared stylesheet
│   ├── nav.js                  # Shared navigation behavior
│   ├── demo.js                 # Demo page: file upload and result rendering
│   ├── config.js               # API base URL configuration
│   ├── app.yaml                # App Engine deployment configuration
│   ├── design.json             # Design tokens (colors, typography, spacing)
│   └── assets/                 # Static plot images (PNG exports from notebook)
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── pca_scatter.png
│       ├── roc_curves.png
│       ├── eda_heatmap_clean.png
│       └── eda_hist_features_light_palette.png
│
├── model/                      # Trained model artifacts
│   ├── vocal_model_lr.joblib   # Deployed: Logistic Regression pipeline (scaler + LR)
│   ├── vocal_model_rf.joblib   # Comparison: Random Forest pipeline
│   └── feature_names.joblib    # Ordered list of feature names expected by the model
│
├── notebooks/
│   └── CS_163_Project.ipynb    # Full ML pipeline: EDA, feature extraction, training, evaluation
│
├── vocal_features.csv          # Extracted feature dataset (output of notebook)
└── README.md
```

---

## Pipeline Overview

The project follows a five-stage pipeline from raw audio to live prediction:

**1. Feature Extraction (notebook)**  
Raw `.wav` clips from the VocalSet dataset are processed using `librosa` and `praat-parselmouth`. Each clip is reduced to a 17-dimensional feature vector: fundamental frequency (f0), spectral centroid, spectral rolloff, harmonic-to-noise ratio (HNR), and 13 MFCC coefficients. Invalid clips (f0 outside 50–500 Hz or HNR below −20 dB) are filtered out. The result is saved to `vocal_features.csv`.

**2. Model Training (notebook)**  
Three classifiers are trained on an 80/20 stratified split: Logistic Regression, Random Forest, and XGBoost. Each is evaluated on both the base 4-feature set and the full 17-feature set. The best models are exported using `joblib` to the `model/` directory.

**3. Inference Service (backend)**  
A FastAPI application wraps the trained Logistic Regression pipeline. It accepts a `.wav` file upload, runs the same extraction code used in training, and returns a JSON prediction. The service is containerized with Docker and deployed to Google Cloud Run.

**4. Prediction Logging (Firestore)**  
After each prediction, the result (label, confidence, extracted features, timestamp) is asynchronously written to a Google Firestore `predictions` collection. This logging is fail-soft: if Firestore is unavailable, the prediction response is still returned to the user.

**5. Frontend (App Engine)**  
A static website built with HTML, CSS, and Vanilla JavaScript presents the project as an academic report with five pages: Homepage, EDA, Methods, Findings, and a Live Demo. The demo page uses the browser Fetch API to call the Cloud Run endpoint and render the prediction result.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker Desktop (only needed to rebuild the backend image)
- A Google Cloud project with Cloud Run, App Engine, and Firestore (Native mode) enabled

### Running the Notebook

1. Clone the repo:
   ```sh
   git clone https://github.com/ZethT/ZethT-Vocal-Affect-Recognition.git
   cd ZethT-Vocal-Affect-Recognition
   ```

2. Install notebook dependencies:
   ```sh
   pip install librosa praat-parselmouth pandas numpy scikit-learn xgboost joblib matplotlib seaborn
   ```

3. Place the VocalSet audio data in `/content/FULL/` (or update paths in the notebook), then run `notebooks/CS_163_Project.ipynb` from top to bottom to reproduce feature extraction, training, and plot generation.

### Running the Backend Locally

```sh
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

The API will be available at `http://localhost:8080`. Update `frontend/config.js` to point to this URL when testing locally.

### Building and Pushing the Docker Image

```sh
docker buildx build --platform linux/amd64 -t gcr.io/<PROJECT_ID>/vocal-affect-api:latest -f backend/Dockerfile . --push
```

---

## System Design

```
┌──────────────────────────────────────────────────────┐
│                   User's Browser                     │
│         Static site served by App Engine             │
│   index / eda / methods / findings / demo pages      │
└────────────────────┬─────────────────────────────────┘
                     │  POST /predict (.wav file)
                     ▼
┌──────────────────────────────────────────────────────┐
│            Google Cloud Run (Docker)                 │
│              FastAPI Inference Service               │
│  features.py → model.py → Logistic Regression        │
└──────────┬───────────────────────────────────────────┘
           │  Async write (fail-soft)
           ▼
┌──────────────────────────────────────────────────────┐
│          Google Firestore (Native mode)              │
│          Collection: predictions                     │
│  Stores label, confidence, features, timestamp       │
└──────────────────────────────────────────────────────┘
```

**Frontend (App Engine):** Hosts the static HTML/CSS/JS website. App Engine serves static files with zero server management. It scales automatically with traffic.

**Inference Service (Cloud Run):** Runs the Docker container that handles audio processing and model inference. Cloud Run scales to zero when idle (cold starts of ~10–15 seconds are expected) and scales up to handle concurrent requests automatically. No infrastructure to manage.

**Cloud Storage (Firestore):** Every successful prediction is logged as a document in the `predictions` Firestore collection. The data could be consumed by a future analytics dashboard or used to retrain the model on real-world inputs.

**Scalability:** Both App Engine and Cloud Run are fully managed and auto-scaling. The inference container is stateless (no session data, model loaded from a file at startup), so Cloud Run can spin up multiple instances in parallel under load without any coordination overhead. The Logistic Regression model is very small (~10 KB) and inference is sub-millisecond, meaning throughput is limited almost entirely by feature extraction time (~0.5–1.5 seconds per clip).

---

## Inference Service

**Location:** `backend/`

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check. Returns `{"status": "ok"}`. |
| `POST` | `/predict` | Accepts a `.wav` file upload. Returns prediction JSON. |

**Input:** A `.wav` audio file (multipart form upload). Recommended: 1–4 seconds of clean singing.

**Output:**
```json
{
  "label": "high_energy",
  "confidence": 0.94,
  "probabilities": {
    "high_energy": 0.94,
    "low_energy": 0.06
  },
  "features": {
    "f0": 312.4,
    "spectral_centroid": 2841.7,
    "spectral_rolloff": 5210.3,
    "hnr": 14.2
  }
}
```

**Key files:**

- `backend/features.py` — extracts all 17 acoustic features from a `.wav` path
- `backend/model.py` — loads `vocal_model_lr.joblib` and runs `pipeline.predict_proba()`
- `backend/main.py` — FastAPI app, validation, error handling, and Firestore logging
- `backend/firestore_client.py` — lazy-initialized Firestore client with fail-soft behavior
- `backend/Dockerfile` — builds a `linux/amd64` image based on `python:3.11-slim`, installs system deps (`libsndfile1`, `cmake`), copies code and model artifact, and starts `uvicorn` on port 8080

**Model artifact:** `model/vocal_model_lr.joblib` — a scikit-learn `Pipeline` containing a `StandardScaler` followed by a `LogisticRegression` classifier, trained on all 17 features.

---

## Cloud Data Storage

**Service:** Google Firestore (Native mode)  
**Project:** `vocalaffect`  
**Collection:** `predictions`

Each document in the `predictions` collection represents one inference request and contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | Timestamp | UTC time of the prediction |
| `label` | String | Predicted class: `high_energy` or `low_energy` |
| `confidence` | Float | Model's probability for the predicted class |
| `probabilities` | Map | Full class probability distribution |
| `f0` | Float | Extracted fundamental frequency (Hz) |
| `spectral_centroid` | Float | Extracted spectral centroid (Hz) |
| `spectral_rolloff` | Float | Extracted spectral rolloff (Hz) |
| `hnr` | Float | Extracted harmonic-to-noise ratio (dB) |

**How it is written:** After each successful prediction, `backend/firestore_client.py` calls `client.collection("predictions").add(doc)`. The Firestore client is lazily initialized on the first prediction request. If initialization or the write fails, the error is logged as a warning and the prediction response is still returned to the user.

**How it could be consumed:** The logged data could be queried from Firestore to analyze prediction distributions over time, identify edge cases where the model is uncertain (low confidence scores), or build a future retraining dataset from real-world uploads.

---

## Model Results

| Model | Base Features (4) | + MFCCs (17) | Gain |
|-------|-------------------|--------------|------|
| Logistic Regression (deployed) | 58.6% | **91.4%** | +32.8 pp |
| Random Forest | ~70.0% | 90.0% | +20.0 pp |
| XGBoost | — | 92.8% | — |

Logistic Regression was selected for deployment over XGBoost (92.8%) because it offers nearly the same accuracy with simpler predictions, clear feature interpretability, and a smaller serving footprint.

---

## Built With

**Languages:** Python · JavaScript · HTML · CSS

**ML / Audio:** scikit-learn · librosa · praat-parselmouth · numpy · pandas · joblib · xgboost

**Backend:** FastAPI · uvicorn · python-multipart

**Cloud:** Google Cloud Run · Google App Engine · Google Firestore · Docker · Google Artifact Registry
