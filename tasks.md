# MVP Build Tasks — Vocal Affect Recognition

Stack: HTML/CSS/Vanilla JS · FastAPI · scikit-learn · librosa · parselmouth · Firestore · Cloud Run · App Engine

> **Deployed model:** Logistic Regression + MFCC pipeline (92.9% accuracy).
> Every page, chart caption, and demo result must say "Logistic Regression" — not Random Forest.
> Random Forest (90.0%) is discussed in Findings as a comparison, but is NOT the live model.

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
  **Done when:** `notebooks/CS_163_Project.ipynb` exists; root is clean.

- [ ] **0.3** Verify `.gitignore` is correct for this project:
  - `*.html` must NOT be ignored (frontend pages need to be tracked)
  - `*.png` must NOT be ignored (chart exports need to be tracked)
  - `model/vocal_model_lr.joblib` must NOT be ignored (needed for Docker build)
  - `.env` must be ignored
  - Only ignore `model/experimental_*.joblib` or `experiments/*.joblib`
  **Done when:** `git check-ignore -v frontend/index.html` returns nothing; `git check-ignore -v .env` returns `.env`.

- [ ] **0.4** Create `backend/requirements.txt` with pinned versions for:
  `fastapi`, `uvicorn`, `librosa`, `numpy`, `scikit-learn`, `joblib`, `praat-parselmouth`, `google-cloud-firestore`, `python-multipart`.
  **Done when:** `pip install -r backend/requirements.txt` completes without error in a fresh venv.

- [ ] **0.5** Create `backend/.env.example` with placeholder keys:
  `GOOGLE_APPLICATION_CREDENTIALS`, `FIRESTORE_PROJECT_ID`, `ENV=dev`.
  **Done when:** File exists; no real secrets are in it; `.env` itself is git-ignored.

---

## Phase 1 — Model Export

- [ ] **1.1** In the Colab notebook, add a final cell that exports the Logistic Regression + MFCC pipeline:
  ```python
  import joblib
  joblib.dump(lr, 'vocal_model_lr.joblib')
  ```
  The `lr` pipeline already includes the `StandardScaler` step — no separate scaler export needed.
  **Done when:** `vocal_model_lr.joblib` is downloadable from Colab.

- [ ] **1.2** Download `vocal_model_lr.joblib` from Colab and save to `model/vocal_model_lr.joblib`.
  **Done when:** File exists locally and is > 0 bytes.

- [ ] **1.3** Create `backend/tests/verify_model.py` — a permanent sanity-check script.
  It should load `model/vocal_model_lr.joblib` and run `predict` on a dummy numpy array of 17 features
  (f0, spectral_centroid, spectral_rolloff, hnr, mfcc_0–mfcc_12).
  Print the prediction and probabilities.
  **Done when:** `python backend/tests/verify_model.py` prints `Prediction: high_energy` or `low_energy` without error.
  Keep this script — it is the fastest way to confirm the model file is intact after any changes.

---

## Phase 2 — FastAPI Backend

### 2A — Feature Extraction Module

- [ ] **2.1** Create `backend/features.py`.
  Add `compute_f0_yin(y, sr)` — copied from notebook, returns float.
  **Done when:** `from features import compute_f0_yin` works without error.

- [ ] **2.2** Add `compute_hnr_praat(file_path)` to `backend/features.py`.
  **Done when:** Calling it on a `.wav` path returns a float.

- [ ] **2.3** Add `extract_features(file_path) -> dict` to `backend/features.py`.
  Returns keys: `f0`, `spectral_centroid`, `spectral_rolloff`, `hnr`, `mfcc_0`–`mfcc_12`.
  **Done when:** Calling it on any local `.wav` file returns a dict with 17 keys, no NaN/Inf.

- [ ] **2.4** Add input validation inside `extract_features`: raise `ValueError` if f0 < 50 or f0 > 500,
  or if centroid/rolloff ≤ 0 (mirrors notebook cleaning logic).
  **Done when:** Passing a silent `.wav` raises `ValueError`, not a 500 crash.

### 2B — Inference Module

- [ ] **2.5** Create `backend/model.py`.
  Load `vocal_model_lr.joblib` using `joblib.load` at module import time.
  **Done when:** `from model import pipeline` works without error.

- [ ] **2.6** Add `predict(features: dict) -> dict` to `backend/model.py`.
  Returns:
  ```json
  {
    "label": "high_energy",
    "confidence": 0.91,
    "probabilities": { "high_energy": 0.91, "low_energy": 0.09 }
  }
  ```
  **Done when:** Calling `predict({...17 features...})` returns a well-formed dict.

### 2C — API

- [ ] **2.7** Create `backend/main.py` with a FastAPI app.
  Add `GET /health` → `{"status": "ok"}`.
  **Done when:** `uvicorn main:app --reload` starts; `curl localhost:8000/health` returns 200.

- [ ] **2.8** Add `POST /predict` to `backend/main.py`.
  Accepts a `.wav` file upload (`UploadFile`), saves to temp file, calls `extract_features`, calls `predict`, returns JSON.
  **Done when:** `curl -F "file=@test.wav" localhost:8000/predict` returns JSON with `label`, `confidence`, `probabilities`, and the extracted features.

- [ ] **2.9** Add error handling: return HTTP 422 with `{"error": "..."}` if `extract_features` raises `ValueError`.
  **Done when:** Uploading a silent or non-vocal `.wav` returns 422, not 500.

- [ ] **2.10** Add CORS middleware to `main.py` allowing all origins.
  **Done when:** A `fetch()` from a local HTML file to `localhost:8000/predict` produces no CORS error.

---

## Phase 3 — Docker + Cloud Run

- [ ] **3.1** Create `backend/Dockerfile`:
  - Base: `python:3.11-slim`
  - Copy `requirements.txt` → run `pip install`
  - Copy app code
  - Copy `../model/vocal_model_lr.joblib` into the image at a known path
  - `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]`
  **Done when:** `docker build -t vocal-backend .` completes without error.

- [ ] **3.2** Run container locally: `docker run -p 8080:8080 vocal-backend`.
  **Done when:** `curl localhost:8080/health` returns `{"status": "ok"}`.

- [ ] **3.3** Test `/predict` through the running Docker container with a real `.wav` file.
  **Done when:** Returns correct prediction JSON from the container (not the dev server).

- [ ] **3.4** Create `backend/.dockerignore` excluding `__pycache__`, `*.pyc`, `.env`, `.venv`.
  **Done when:** Image builds cleanly; size is reasonable (check with `docker images`).

- [ ] **3.5** Push the Docker image to Google Artifact Registry.
  Tag: `gcr.io/[PROJECT_ID]/vocal-backend:v1`
  **Done when:** `docker push` succeeds; image is visible in GCP console.

- [ ] **3.6** Deploy to Cloud Run:
  ```
  gcloud run deploy vocal-backend \
    --image gcr.io/[PROJECT_ID]/vocal-backend:v1 \
    --allow-unauthenticated \
    --region us-central1
  ```
  **Done when:** Cloud Run gives a public URL; `curl [URL]/health` returns 200.

- [ ] **3.7** Test `/predict` at the live Cloud Run URL with a real `.wav` file.
  **Done when:** Live URL returns correct prediction JSON.

- [ ] **3.8** Create `frontend/config.js` with `const API_BASE_URL = "https://[cloud-run-url]"`.
  **Done when:** `config.js` has the URL; no URL is hard-coded anywhere else.

---

## Phase 4 — Frontend Pages

All pages share `frontend/style.css`. Nav is a shared `<nav>` block repeated in each file.

- [ ] **4.1** Create `frontend/style.css` with:
  - System-ui font stack
  - CSS variables: `--color-belt: #2196F3`, `--color-breathy: #FF9800`
  - Responsive centered container (max-width: 1100px)
  - Nav bar, heading, and body text styles
  **Done when:** Opening any page in a browser shows a styled layout with no unstyled text.

- [ ] **4.2** Export the following plots from the notebook as PNG files, save to `frontend/assets/`:
  - `confusion_matrix.png`
  - `feature_importance.png`
  - `pca_scatter.png`
  - `roc_curves.png`
  - `model_accuracy_comparison.png`
  **Done when:** All 5 PNGs exist in `frontend/assets/`; each displays correctly in a browser `<img>` tag.

- [ ] **4.3** Create `frontend/index.html` — Landing Page.
  Sections:
  1. Hero: project title + one-sentence description
  2. What we built: 3-bullet summary
  3. Two embedded chart images (`confusion_matrix.png`, `feature_importance.png`) with captions
  4. CTA button → `demo.html`
  **Done when:** Page renders; images load; nav links to all pages work.

- [ ] **4.4** Create `frontend/objective.html` — Project Objective Page.
  Sections:
  1. Problem statement (belt vs. breathy, arousal mapping to mood)
  2. Dataset: VocalSet — source, size, what it contains
  3. Goal: classify vocal affect to improve music recommendation
  4. Russell's Circumplex: brief explanation + static diagram image
  **Done when:** Page renders fully; no broken links or missing images.

- [ ] **4.5** Create `frontend/eda.html` — Exploratory Data Analysis Page.
  Sections:
  1. Dataset overview: sample counts per class, feature list
  2. Embed `pca_scatter.png` with written interpretation
  3. Embed `roc_curves.png` with written interpretation
  4. Key EDA insight: HNR artifact filtering note; centroid/rolloff correlation (r = 0.89)
  **Done when:** Page renders; all charts load; written interpretations are present (not placeholder).

- [ ] **4.6** Create `frontend/methods.html` — Analytical Methods Page.
  Sections:
  1. Feature extraction table: f0, spectral_centroid, spectral_rolloff, HNR, MFCCs — what each measures
  2. Models compared: Logistic Regression, Random Forest, XGBoost — one paragraph each
  3. Note which model is deployed (Logistic Regression, 92.9%)
  **Done when:** Page renders; no placeholder text; model descriptions are accurate.

- [ ] **4.7** Create `frontend/findings.html` — Major Findings Page.
  Sections:
  1. Accuracy results table (all 4 model variants — basic vs. +MFCC for LR and RF)
  2. Embed `model_accuracy_comparison.png`
  3. Embed `confusion_matrix.png` and `feature_importance.png`
  4. Key insight bullets: MFCC features caused the accuracy jump; deployed model is Logistic Regression
  **Done when:** Page renders; results table matches notebook output; deployed model is named correctly.

- [ ] **4.8** Add consistent `<nav>` to all five pages: Home · Objective · EDA · Methods · Findings · Demo.
  Active page link is visually highlighted.
  **Done when:** Nav is present on all pages; no dead links.

---

## Phase 5 — Interactive Demo Page

- [ ] **5.1** Create `frontend/demo.html` with:
  - `<input type="file" accept=".wav">` file picker
  - "Classify" button
  - Empty result section (hidden by default)
  **Done when:** Page renders; file picker opens on click.

- [ ] **5.2** Create `frontend/demo.js` (loaded via `<script src="config.js">` then `<script src="demo.js">`).
  On "Classify" click: read the selected `.wav`, POST as `FormData` to `API_BASE_URL/predict`.
  **Done when:** Browser devtools Network tab shows the POST request being sent.

- [ ] **5.3** Display prediction result:
  - Predicted class label with plain-English description (High Energy = belt / Low Energy = breathy)
  - Confidence percentage
  - Two probability bars colored with CSS variables (`--color-belt`, `--color-breathy`)
  **Done when:** After uploading a `.wav`, result section appears with correct values from the live API.

- [ ] **5.4** Add a collapsible `<details>` section showing extracted features:
  `f0`, `spectral_centroid`, `spectral_rolloff`, `hnr` as a small table.
  **Done when:** "Show extracted features" expands the table with real values (not mock data).

- [ ] **5.5** Add a CSS-only loading spinner shown while the request is in-flight; hide it on response.
  **Done when:** Spinner appears on submit; disappears when result renders or error shows.

- [ ] **5.6** Add error handling in `demo.js`: non-200 response or network failure shows a readable
  error message in the result area.
  **Done when:** Uploading a non-audio file shows an error message; no JS console crash.

---

## Phase 6 — App Engine Deployment (Frontend)

- [ ] **6.1** Create `frontend/app.yaml` with explicit static file handlers:
  ```yaml
  runtime: python39
  handlers:
    - url: /assets
      static_dir: assets
    - url: /(.+\.(css|js))
      static_files: \1
      upload: .+\.(css|js)
    - url: /(.+\.html)
      static_files: \1
      upload: .+\.html
    - url: /
      static_files: index.html
      upload: index.html
  ```
  ⚠️ Flag for testing: App Engine static routing can be finicky. Test each page URL explicitly after deploy.
  **Done when:** YAML syntax is valid (`gcloud app deploy --dry-run` passes).

- [ ] **6.2** Deploy frontend: `gcloud app deploy frontend/app.yaml`.
  **Done when:** `gcloud app browse` opens the landing page in a browser.

- [ ] **6.3** Test all 6 pages at the live App Engine URL: index, objective, eda, methods, findings, demo.
  **Done when:** All pages load with styles, images, and working nav.

- [ ] **6.4** Test the full demo flow at the live URL:
  Upload a `.wav` → prediction appears on page → confirm API call goes to Cloud Run URL (not localhost).
  **Done when:** End-to-end flow works without any localhost references.

---

## Phase 7 — Firestore Integration

> Firestore is moved here (after the demo works) so a cloud DB issue never blocks the demo.

- [ ] **7.1** Create a Firestore database (Native mode) in the GCP project. Collection: `predictions`.
  **Done when:** Collection is visible in Firebase console.

- [ ] **7.2** Create `backend/firestore_client.py`.
  Initialize `google.cloud.firestore.Client()`.
  Add `save_prediction(data: dict) -> str` — writes timestamp, label, confidence, probabilities,
  f0, spectral_centroid, spectral_rolloff, hnr; returns the document ID.
  **Done when:** Calling it locally creates a document in Firestore.

- [ ] **7.3** Call `save_prediction` inside the `/predict` endpoint in `main.py` after a successful inference.
  Wrap in `try/except` so a Firestore failure logs a warning but does NOT fail the prediction response.
  **Done when:** A `/predict` call creates a new Firestore document AND still returns the prediction
  even when Firestore credentials are missing (just logs a warning).

- [ ] **7.4** Redeploy the backend Docker image with Firestore env vars set in Cloud Run.
  **Done when:** After a live demo prediction, document appears in Firestore console.

---

## Phase 8 — README + Repo Polish

- [ ] **8.1** Create `README.md` with:
  1. Project overview (2–3 sentences)
  2. Repo structure (folder tree)
  3. Local setup: venv, `pip install -r backend/requirements.txt`, `uvicorn main:app --reload`
  4. How to run the notebook (Colab link or local instructions)
  **Done when:** A new contributor can clone and run locally using only the README.

- [ ] **8.2** Add to `README.md`:
  5. Pipeline explanation: audio file → feature extraction → LR model → prediction JSON
  6. System design diagram (copy from `architecture.md`)
  7. Inference service explanation: what `/predict` accepts and returns
  8. Cloud storage explanation: what Firestore stores and why it satisfies the rubric
  9. Deployment instructions: Cloud Run + App Engine commands with placeholders for project ID
  **Done when:** All rubric repo requirements are addressed.

- [ ] **8.3** Run `git check-ignore -v frontend/*.html frontend/assets/*.png model/vocal_model_lr.joblib`
  and confirm none of them are ignored.
  **Done when:** Command returns no output for those files (nothing is being ignored).

- [ ] **8.4** Final cleanup: remove debug `print` statements from backend, remove any `console.log`
  from `demo.js`, verify no placeholder text remains on any page.
  **Done when:** All pages look clean; backend logs only real errors.
