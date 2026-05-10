# MVP Build Tasks — Vocal Affect Recognition

Stack: HTML/CSS/Vanilla JS · FastAPI · scikit-learn · librosa · parselmouth · Firestore · Cloud Run · App Engine

**Rubric checklist (must hit all 8):**
- [ ] Landing page with visualizations
- [ ] Project Objective page
- [ ] Analytical Methods page
- [ ] Major Findings page
- [ ] At least one interactive feature
- [ ] Hosted on Google App Engine
- [ ] Data stored in Firestore
- [ ] ML inference on Cloud Run (Docker)

---

## Phase 0 — Project Structure

- [ ] **0.1** Create the top-level folder layout:
  ```
  /backend       ← FastAPI inference service
  /frontend      ← HTML/CSS/JS website
  /model         ← exported model artifacts
  /notebooks     ← move CS_163_Project.ipynb here
  ```
  **Done when:** `ls` shows all four folders exist.

- [ ] **0.2** Move `CS_163_Project.ipynb` into `/notebooks`.
  **Done when:** `notebooks/CS_163_Project.ipynb` exists, root is clean.

- [ ] **0.3** Create `backend/requirements.txt` with pinned versions for:
  `fastapi`, `uvicorn`, `librosa`, `numpy`, `scikit-learn`, `joblib`, `praat-parselmouth`, `google-cloud-firestore`, `python-multipart`.
  **Done when:** `pip install -r backend/requirements.txt` completes without error in a fresh venv.

- [ ] **0.4** Create `backend/.env.example` with placeholder keys:
  `GOOGLE_APPLICATION_CREDENTIALS`, `FIRESTORE_PROJECT_ID`, `ENV=dev`.
  **Done when:** File exists; no real secrets are in it.

---

## Phase 1 — Model Export

- [ ] **1.1** In the Colab notebook, add a final cell that does:
  ```python
  import joblib
  joblib.dump(lr, 'vocal_model_lr.joblib')   # best accuracy: 92.9%
  joblib.dump(lr.named_steps['scaler'], 'scaler.joblib')
  ```
  Export the **Logistic Regression + MFCC pipeline** (highest accuracy model).
  **Done when:** `vocal_model_lr.joblib` is downloadable from Colab.

- [ ] **1.2** Download `vocal_model_lr.joblib` from Colab and save to `/model/vocal_model_lr.joblib`.
  **Done when:** File exists locally and is > 0 bytes.

- [ ] **1.3** Write a one-off Python script `model/verify_model.py` that loads the model and runs a predict on a dummy numpy array of 17 features (f0, centroid, rolloff, hnr, mfcc_0–12).
  **Done when:** Script prints `Prediction: high_energy or low_energy` without error.

- [ ] **1.4** Delete `model/verify_model.py` after it passes — it was only for verification.
  **Done when:** Only the `.joblib` file remains in `/model`.

---

## Phase 2 — FastAPI Backend

### 2A — Feature Extraction Module

- [ ] **2.1** Create `backend/features.py`.
  Add `compute_f0_yin(y, sr)` — copied from notebook, returns float.
  **Done when:** Unit import works: `from features import compute_f0_yin`.

- [ ] **2.2** Add `compute_hnr_praat(file_path)` to `backend/features.py`.
  **Done when:** Function exists; calling it on a `.wav` path returns a float.

- [ ] **2.3** Add `extract_features(file_path) -> dict` to `backend/features.py`.
  Returns keys: `f0`, `spectral_centroid`, `spectral_rolloff`, `hnr`, `mfcc_0`–`mfcc_12`.
  **Done when:** Calling it on any local `.wav` file returns a dict with 17 keys, no NaN/Inf.

- [ ] **2.4** Add input validation inside `extract_features`: raise `ValueError` if f0 < 50 or f0 > 500, or if centroid/rolloff <= 0 (mirrors notebook cleaning logic).
  **Done when:** Passing a silent `.wav` raises `ValueError`, not a crash.

### 2B — Inference Module

- [ ] **2.5** Create `backend/model.py`.
  Load `vocal_model_lr.joblib` using `joblib.load` at module import time.
  **Done when:** `from model import pipeline` works without error.

- [ ] **2.6** Add `predict(features: dict) -> dict` to `backend/model.py`.
  Returns `{ "label": str, "confidence": float, "probabilities": {"high_energy": float, "low_energy": float} }`.
  **Done when:** Calling `predict({...17 features...})` returns a well-formed dict.

### 2C — API Endpoint

- [ ] **2.7** Create `backend/main.py` with a FastAPI app.
  Add `GET /health` that returns `{"status": "ok"}`.
  **Done when:** `uvicorn main:app --reload` starts; `curl localhost:8000/health` returns 200.

- [ ] **2.8** Add `POST /predict` to `backend/main.py`.
  Accepts a `.wav` file upload (`UploadFile`), saves to a temp file, calls `extract_features`, calls `predict`, returns the prediction JSON.
  **Done when:** `curl -F "file=@test.wav" localhost:8000/predict` returns a JSON response with `label`, `confidence`, `probabilities`.

- [ ] **2.9** Add `POST /predict` error handling: return HTTP 422 with a message if `extract_features` raises `ValueError`.
  **Done when:** Uploading a silent or invalid file returns `{"error": "..."}` with status 422, not a 500.

- [ ] **2.10** Add CORS middleware to `main.py` to allow requests from `*` (permissive for academic use).
  **Done when:** A `fetch()` from a local HTML file to `localhost:8000/predict` does not produce a CORS error.

---

## Phase 3 — Docker + Cloud Run

- [ ] **3.1** Create `backend/Dockerfile`:
  - Base: `python:3.11-slim`
  - Copy `requirements.txt`, run `pip install`
  - Copy app code and model file
  - Expose port 8080
  - `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]`
  **Done when:** `docker build -t vocal-backend .` completes without error.

- [ ] **3.2** Run the Docker container locally:
  `docker run -p 8080:8080 vocal-backend`
  **Done when:** `curl localhost:8080/health` returns `{"status": "ok"}`.

- [ ] **3.3** Test `/predict` through the Docker container with a real `.wav` file.
  **Done when:** Returns correct prediction JSON from the container.

- [ ] **3.4** Create `backend/.dockerignore` excluding `__pycache__`, `*.pyc`, `.env`, `.venv`.
  **Done when:** Docker build image size is reasonable (< 2 GB).

- [ ] **3.5** Push the Docker image to Google Artifact Registry.
  Tag: `gcr.io/[PROJECT_ID]/vocal-backend:v1`
  **Done when:** `docker push` succeeds; image visible in GCP console.

- [ ] **3.6** Deploy to Cloud Run:
  `gcloud run deploy vocal-backend --image gcr.io/[PROJECT_ID]/vocal-backend:v1 --allow-unauthenticated --region us-central1`
  **Done when:** Cloud Run gives a public URL; `curl [URL]/health` returns 200.

- [ ] **3.7** Test `/predict` at the live Cloud Run URL with a real `.wav` file.
  **Done when:** Live URL returns correct prediction JSON.

- [ ] **3.8** Copy the Cloud Run service URL into `frontend/config.js` as `API_BASE_URL`.
  **Done when:** `config.js` has the URL; no URL is hard-coded in other JS files.

---

## Phase 4 — Firestore Integration

- [ ] **4.1** Create a Firestore database (Native mode) in the GCP project.
  Collection: `predictions`.
  **Done when:** Collection visible in Firebase console.

- [ ] **4.2** Add `backend/firestore_client.py`.
  Initialize Firestore client using `google.cloud.firestore.Client()`.
  **Done when:** `from firestore_client import db` does not throw on import when `GOOGLE_APPLICATION_CREDENTIALS` is set.

- [ ] **4.3** Add `save_prediction(data: dict) -> str` to `firestore_client.py`.
  Writes `{ timestamp, label, confidence, probabilities, f0, spectral_centroid, spectral_rolloff, hnr }` to `predictions` collection.
  Returns the Firestore document ID.
  **Done when:** Calling it locally creates a real document in Firestore.

- [ ] **4.4** Call `save_prediction` inside the `/predict` endpoint in `main.py` (after a successful inference).
  **Done when:** After a `/predict` call, a new document appears in the Firestore console.

- [ ] **4.5** Make Firestore writes non-blocking: wrap in `try/except` so a Firestore failure does NOT fail the prediction response.
  **Done when:** Disabling Firestore credentials still returns a valid prediction (just logs a warning).

---

## Phase 5 — Frontend Pages

All pages share a single `frontend/style.css` and `frontend/nav.html` (included via JS or repeated).

- [ ] **5.1** Create `frontend/style.css` with:
  - Clean sans-serif font (system-ui or Inter via Google Fonts)
  - Consistent color palette (use `--color-belt: #2196F3` and `--color-breathy: #FF9800` from notebook)
  - Responsive max-width container (1100px centered)
  - Nav bar styles
  **Done when:** Opening any page in a browser shows a styled nav and readable body text.

- [ ] **5.2** Create `frontend/index.html` — Landing Page.
  Sections:
  1. Hero: project title + one-sentence description
  2. What we built (3 bullet points)
  3. Two static chart images (embed PNGs exported from notebook): confusion matrix + feature importance
  4. CTA button → `demo.html`
  **Done when:** Page opens in browser, images load, link to demo works.

- [ ] **5.3** Export the following plots from the notebook as PNG files, save to `frontend/assets/`:
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `pca_scatter.png`
  - `roc_curves.png`
  - `model_accuracy_comparison.png`
  **Done when:** All 5 PNGs exist in `frontend/assets/` and display correctly in a browser `<img>` tag.

- [ ] **5.4** Create `frontend/objective.html` — Project Objective Page.
  Sections:
  1. Problem statement (belt vs. breathy, arousal mapping)
  2. Dataset: VocalSet — who made it, how many samples, what it contains
  3. Goal: classify vocal affect to improve music recommendation
  4. Link to Russell's Circumplex (brief explanation, static diagram image)
  **Done when:** Page renders fully; no broken links.

- [ ] **5.5** Create `frontend/methods.html` — Analytical Methods Page.
  Sections:
  1. Feature extraction table (f0, centroid, rolloff, HNR, MFCCs — what each measures)
  2. Models compared (Logistic Regression, Random Forest, XGBoost — one paragraph each)
  3. Embed `pca_scatter.png` with caption
  4. Embed `roc_curves.png` with caption
  **Done when:** Page renders; all images load; no placeholder text remains.

- [ ] **5.6** Create `frontend/findings.html` — Major Findings Page.
  Sections:
  1. Accuracy results table (all 4 model variants)
  2. Embed `model_accuracy_comparison.png`
  3. Embed `confusion_matrix.png` and `feature_importance.png`
  4. Key insight bullets: MFCC features were the critical jump; HNR distinguishes belt from breathy
  **Done when:** Page renders; findings are clearly written; all images load.

- [ ] **5.7** Add a consistent `<nav>` to all four pages linking: Home · Objective · Methods · Findings · Demo.
  **Done when:** Nav is present on all pages; active page link is visually highlighted.

---

## Phase 6 — Interactive Demo Page

- [ ] **6.1** Create `frontend/demo.html` with a file upload `<input type="file" accept=".wav">` and a "Classify" button.
  **Done when:** Page renders; file picker opens on click; button is visible.

- [ ] **6.2** Create `frontend/demo.js`.
  On "Classify" click: read the selected `.wav` file, POST it as `FormData` to `API_BASE_URL/predict` using `fetch()`.
  **Done when:** Browser devtools shows the POST request being sent.

- [ ] **6.3** Display the result after a successful response:
  - Predicted class (belt = High Energy / breathy = Low Energy)
  - Confidence percentage
  - Probability bar for each class (two `<div>` bars, colored with CSS variables)
  **Done when:** After uploading a `.wav`, the result section appears with correct values.

- [ ] **6.4** Display extracted features in a collapsible `<details>` section:
  Show `f0`, `spectral_centroid`, `spectral_rolloff`, `hnr` as a small table.
  **Done when:** Clicking "Show extracted features" reveals the table with real values.

- [ ] **6.5** Add a loading spinner (CSS only) shown while the request is in-flight.
  **Done when:** Spinner appears on submit; disappears when result renders.

- [ ] **6.6** Add error handling in `demo.js`: if the API returns non-200 or network fails, show a user-friendly error message in the result area.
  **Done when:** Uploading a non-audio file shows an error message (not a JS console crash).

---

## Phase 7 — App Engine Deployment (Frontend)

- [ ] **7.1** Create `frontend/app.yaml`:
  ```yaml
  runtime: python39
  handlers:
    - url: /(.*)
      static_files: \1
      upload: (.*)
  ```
  **Done when:** File exists with correct YAML syntax.

- [ ] **7.2** Deploy frontend to App Engine:
  `gcloud app deploy frontend/app.yaml`
  **Done when:** `gcloud app browse` opens the landing page in a browser.

- [ ] **7.3** Verify all 5 pages load correctly at the App Engine URL.
  **Done when:** index, objective, methods, findings, demo — all load with styles and images.

- [ ] **7.4** Test the full demo flow end-to-end at the live App Engine URL:
  Upload a `.wav` → get prediction → check Firestore console for the saved document.
  **Done when:** Prediction appears on page AND document appears in Firestore.

---

## Phase 8 — README + Repo Polish

- [ ] **8.1** Create `README.md` with sections:
  1. Project overview (2–3 sentences)
  2. Repo structure (folder tree)
  3. Local setup instructions (backend venv, pip install, run uvicorn)
  4. How to run the notebook
  **Done when:** A new contributor could clone and run locally using only the README.

- [ ] **8.2** Add to `README.md`:
  5. Pipeline explanation (data → features → model → prediction)
  6. System design diagram (copy architecture from `architecture.md`)
  7. Inference service explanation (what `/predict` does)
  8. Cloud storage explanation (what is stored in Firestore and why)
  9. Deployment instructions (Cloud Run + App Engine commands)
  **Done when:** All rubric repo requirements are covered.

- [ ] **8.3** Verify `.gitignore` covers: `.env`, `*.joblib`, `__pycache__`, `.venv`, `node_modules`, `*.DS_Store`.
  **Done when:** `git status` does not show any secrets or venv files as untracked.

- [ ] **8.4** Final commit: clean up any debug print statements, remove TODO comments, verify no placeholder text remains on any page.
  **Done when:** All pages look production-ready; no `console.log` left in `demo.js`.
