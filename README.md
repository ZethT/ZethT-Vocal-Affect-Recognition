<a name="readme-top"></a>
<br />
<div align="center">
  <h3 align="center">Vocal Affect Recognition</h3>

  <p align="center">
    An automated machine learning pipeline for classifying vocal audio excerpts into high and low energy categories using Digital Signal Processing (DSP) and ensemble modeling.
    <br />
    <a href="https://vocalaffect.uc.r.appspot.com/index.html"><strong>Explore the website »</strong></a>
    <br />
    <br />
  </p>
</div>

## About The Project

This project addresses the challenge of automatically identifying the "energy" of a vocal performance using machine learning. By analyzing the **VocalSet** dataset—a collection of professional singer recordings—the pipeline transforms raw audio into a feature-rich dataset to distinguish between high-intensity and low-intensity vocalizations.

**The Classification Logic:**
* **High Energy:** **Belt**
* **Low Energy:** **Breathy**

By leveraging **Mel-Frequency Cepstral Coefficients (MFCCs)** alongside traditional spectral features (F0, Spectral Centroid, Spectral Rolloff, HNR), the model successfully captures the timbral nuances that distinguish forceful singing from breathy or fry-based vocalizations.

## Directory Structure

```text
.
├── backend/        FastAPI inference service + Dockerfile (deployed to Cloud Run)
├── frontend/       Static website — HTML/CSS/JS (deployed to App Engine)
├── model/          Trained scikit-learn models (.joblib artifacts)
├── notebooks/      EDA + model training notebook (CS_163_Project.ipynb)
├── data/           Processed feature dataset (vocal_features.csv)
└── docs/           Onboarding, EDA notes, and original assignment deliverables
```

## Pipeline

The project flows end-to-end from raw audio to a live web demo:

1. **Data collection** — VocalSet `.wav` recordings (belt + breathy classes).
2. **Feature extraction** — `notebooks/CS_163_Project.ipynb` computes 17 acoustic features per clip (F0, Spectral Centroid, Spectral Rolloff, HNR, 13 MFCCs).
3. **Dataset** — Extracted features are written to `data/vocal_features.csv`.
4. **Training** — The same notebook trains Logistic Regression, Random Forest, and XGBoost on an 80/20 stratified split; the best model is serialized to `model/vocal_model_lr.joblib`.
5. **Inference** — `backend/` wraps the model in a FastAPI service, containerized via `backend/Dockerfile` for Cloud Run.
6. **Website** — `frontend/` is a static site served by App Engine; the demo page uploads a `.wav` to the inference service and renders the prediction.

## System Design

```text
   Browser
      │
      ▼
   App Engine  ──────── serves static HTML/CSS/JS from frontend/
      │
      │  (demo.html fetch → /predict)
      ▼
   Cloud Run  ───────── FastAPI (Docker) — feature extraction + sklearn inference
      │
      ▼
   Firestore  ───────── prediction logs
```

**Scalability.** The frontend is static, so App Engine serves it from edge cache with no per-request compute. The inference service is stateless: the model is loaded once at container startup (`backend/model.py`) and each request runs feature extraction + prediction in isolation, which lets Cloud Run autoscale instances horizontally with traffic. Storage is decoupled — Firestore writes are best-effort and never block a response.

## Built With

### Languages
* ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
* ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
* ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
* ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)

### Libraries & Frameworks
* ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) - Inference API
* ![Uvicorn](https://img.shields.io/badge/Uvicorn-2A9D8F?style=flat) - ASGI server
* ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) - Machine Learning Model
* ![Librosa](https://img.shields.io/badge/Librosa-lightgrey?style=flat) - Audio Feature Extraction
* ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) - Data Manipulation
* ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) - Numerical Processing

### Infrastructure & Deployment
* ![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=flat&logo=google-cloud&logoColor=white) - Hosting (App Engine + Cloud Run)
* ![Firebase](https://img.shields.io/badge/Firebase-FFCA28?style=flat&logo=firebase&logoColor=black) - Firestore Database
* ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white) - Containerization

## Getting Started

### Prerequisites

Python 3.11+ is required (the deployed container is built on `python:3.11-slim`).

### Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/ZethT/ZethT-Vocal-Affect-Recognition.git
   cd ZethT-Vocal-Affect-Recognition
   ```

2. Install backend dependencies:
   ```sh
   cd backend
   pip install -r requirements.txt
   ```

### Running the inference service locally

From the `backend/` directory:

```sh
uvicorn main:app --host 0.0.0.0 --port 8080
```

This matches the command in `backend/Dockerfile`.

### Running via Docker

The Dockerfile is built **from the repo root** so it can include `model/` artifacts:

```sh
docker build -f backend/Dockerfile -t vocal-backend .
docker run -p 8080:8080 vocal-backend
```

### Running the notebook

Open `notebooks/CS_163_Project.ipynb` in Jupyter or Google Colab. The notebook handles feature extraction, EDA, and model training; it writes `data/vocal_features.csv` and the `.joblib` artifacts in `model/`.

## Results

Our analysis proved that MFCCs are critical for energy classification. Adding MFCCs to the baseline spectral features boosted the Logistic Regression model's accuracy from 58.57% to 92.86%.

| Model | Features | Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | Basic Spectral | 58.57% |
| Random Forest | Basic Spectral | 70.00% |
| Logistic Regression | Basic + MFCCs | 92.86% |
| Random Forest | Basic + MFCCs | 90.00% |

The final Logistic Regression (+ MFCC) model is the recommended choice for production due to its high accuracy and lower computational overhead.

## Roadmap
- [x] Initial DSP Feature Extraction (F0, Centroid, HNR)
- [x] Comparative Analysis of ML Architectures
- [x] Timbral Optimization with MFCCs
- [ ] Implement cross-validation for more robust error estimation
- [ ] Build a web-based UI for real-time `.wav` classification
