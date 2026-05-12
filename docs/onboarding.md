# Onboarding: Vocal Affect Recognition Project

**CS 163 Sec 01 — Group 16**
**Members:** Tanzil Ahmed, James Kim, Zeth Tang
**Last updated:** March 22, 2026

---

## Elevator Pitch (2 minutes)

> Have you ever been listening to a "Chill Vibes" playlist on Spotify, and suddenly a high-energy belting track comes on that completely kills the mood? That happens because current recommendation systems tag songs by genre and listening history — they don't actually understand the *emotional delivery* of the voice.
>
> Our project fixes that. We're building a **Mood Tagger** — an automated system that listens to *how* a singer is singing, not *what* they're singing, and classifies the track's emotional energy based on vocal technique.
>
> Here's the science behind it. When a singer belts — think Adele hitting a power note — the voice produces strong, periodic harmonics. The Harmonic-to-Noise Ratio is high, the spectral energy is bright and concentrated. When a singer goes breathy — think Billie Eilish whispering into the mic — the voice introduces turbulent airflow across the vocal folds, which shows up as broadband noise. The HNR drops, the spectral profile spreads out. And these aren't just two extremes — forte singing, vibrato, vocal fry, pianissimo all fall on this energy spectrum.
>
> We're training on ~2,550 recordings from VocalSet covering 10 different vocal techniques, grouped into high-energy (belt, forte, vibrato) and low-energy (breathy, pianissimo, vocal fry). We extract four acoustic features and build classifiers that can tell the difference.
>
> But here's the real question: does vocal technique actually predict emotion? To prove it, we take our technique-trained model and run it on RAVDESS — a dataset of actors singing in different emotions. If our model says angry singing is high-energy and calm singing is low-energy, *without ever seeing emotion labels*, we've proven the link.
>
> Then we go to real music. We take songs from the Free Music Archive, use Spleeter to strip out the vocals, and run our classifier. Does a metal track get tagged high-energy? Does indie folk get tagged low-energy?
>
> Finally, we add valence — positive vs negative emotion — using Music2Emo, a pre-trained model. Our vocal analysis gives us the energy axis, Music2Emo gives us the positivity axis. Together, we map any song onto Russell's Circumplex: happy, angry, sad, or calm.
>
> The end goal is a web dashboard where you upload any song, we strip out the vocals, and tell you the mood — not just "high energy" but the full emotional quadrant, with an explanation of *why* based on the physics of the human voice.

---

## What Are We Building?

A **Mood Tagger** that classifies a song's emotional state using two components:
- **Arousal (energy level)** — our model, built from scratch, using vocal acoustic features. This is our core contribution — interpretable, explainable, grounded in vocal science.
- **Valence (positive/negative)** — Music2Emo, a pre-trained model. Augments our arousal with the dimension vocal technique alone can't capture.

Together these map onto **Russell's Circumplex Model of Affect** — the standard 2D emotion framework:

```
                High Arousal (OUR MODEL)
                        │
             Angry ─────┼───── Happy
                        │
  Low Valence ──────────┼────────── High Valence
  (Music2Emo)           │           (Music2Emo)
                        │
              Sad  ─────┼───── Calm
                        │
                Low Arousal (OUR MODEL)
```

**Final deliverable:** A web dashboard (Dash/Plotly) where a user uploads a song → Spleeter isolates vocals → our model classifies arousal → Music2Emo classifies valence → output: mood quadrant + explanation.

---

## Project Pipeline (End to End)

```
1. VocalSet (training data)       → ~2,550 recordings across 10 techniques
2. Feature Extraction             → Extract 4 acoustic features per recording
3. EDA                            → Explore data, find patterns (DONE)
4. Model Training                 → Train arousal classifiers on VocalSet (CURRENT)
5. RAVDESS Validation             → Prove technique = emotion (no retraining)
6. Spleeter + FMA Validation      → Test on real-world music
7. Music2Emo Valence Integration  → Add positive/negative dimension
8. Dashboard                      → Web app for full mood classification
```

---

## Where We Are Now

| Phase | Status |
|-------|--------|
| Dataset collection (VocalSet) | DONE |
| Feature extraction (f0, centroid, rolloff, HNR) | DONE (for belt/breathy — needs expansion to 10 techniques) |
| EDA Summary (submitted) | DONE |
| Analysis & Visualization Plan (submitted) | DONE |
| Expand to 10 techniques | NEXT |
| Model development | NEXT |
| RAVDESS validation | PLANNED |
| FMA + Spleeter validation | PLANNED |
| Music2Emo integration | PLANNED |
| Dashboard | PLANNED |

---

## The Datasets

### VocalSet (Training Data)
- **Source:** [VocalSet](https://zenodo.org/record/1193957) — academic singing voice dataset
- **Total:** 3,613 files across 18 techniques
- **What we use:** 10 techniques grouped into two energy classes (~2,550 files):

| Energy Level | Techniques | Sample Count |
|-------------|-----------|-------------|
| **High energy** | belt (205), forte (100), fast_forte (394), slow_forte (395), vibrato (255) | ~1,350 |
| **Low energy** | breathy (200), pp (100), slow_piano (397), vocal_fry (198), spoken (20) | ~915 |

- **Location:** Google Shared Drive → `CS 163 Project/`
- **Current CSV:** `vocal_features.csv` (405 rows — belt/breathy only, needs re-extraction for expanded set)

### RAVDESS (Emotion Validation Data)
- **Source:** [RAVDESS](https://zenodo.org/records/1188976) — emotional speech and song database
- **What it is:** 24 actors singing two sentences in 5 emotions (calm, happy, sad, angry, fearful) × 2 intensities
- **Purpose:** Validate that our technique-trained model predicts emotional arousal. **No retraining** — we run our VocalSet model directly on RAVDESS.
- **Emotion → Energy mapping:**
  - angry, happy → expect high energy
  - calm, sad → expect low energy
  - fearful → ambiguous (could go either way)

### FMA (Real-World Validation Data)
- **Source:** [Free Music Archive](https://github.com/mdeff/fma)
- **Purpose:** Test on real songs via Spleeter vocal isolation
- **Pipeline:** FMA track → Spleeter → isolated vocals → extract features → classify
- **Validation:** Check predictions against genre expectations (metal = high energy, folk = low energy)

### Music2Emo (Valence Augmentation)
- **Source:** [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) — pre-trained emotion recognition
- **Purpose:** Provides valence (positive/negative) that our vocal features can't capture
- **Usage:** Run on FMA tracks alongside our arousal model → full circumplex mapping

---

## The 4 Acoustic Features

Every audio file gets reduced to these 4 numbers:

| Feature | What It Measures | Library | Key Finding from EDA |
|---------|-----------------|---------|---------------------|
| **f₀ (Fundamental Frequency)** | Pitch of the voice in Hz | `librosa` (YIN algorithm) | Bimodal distribution — likely a gender confound (male vs. female singers have different pitch ranges) |
| **Spectral Centroid** | "Brightness" — where the center of spectral energy sits | `librosa` | Highly correlated with Spectral Rolloff (r = 0.89) — redundant |
| **Spectral Rolloff** | Frequency below which 85% of spectral energy lies | `librosa` | Highly correlated with Spectral Centroid — we use PCA to handle this |
| **HNR (Harmonic-to-Noise Ratio)** | Voice clarity vs. noise, in dB | `parselmouth` (Praat) | **Most discriminative feature.** Belt mean: −5.93 dB, Breathy mean: −8.31 dB. Breathy singing introduces more broadband noise, lowering HNR. |

### Data Quality Notes
- HNR range is extreme: −86 to +24 dB. Values below −20 dB are Praat extraction artifacts
- After filtering HNR ≥ −20 dB: **349 clean samples** (from original 405 belt/breathy set)
- No missing values in the 4 features after extraction
- 8 duplicate rows found and dropped during preprocessing

---

## Tech Stack

| Purpose | Tools |
|---------|-------|
| Audio processing | `librosa` (pitch, spectral features), `parselmouth` / Praat (HNR) |
| Vocal isolation | `spleeter` (Deezer) — separates vocals from instrumentals |
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn` (PCA, Logistic Regression, Random Forest), `xgboost` |
| Interpretability | `shap` (SHAP values for XGBoost) |
| Valence prediction | `music2emo` (pre-trained arousal + valence) |
| Visualization | `seaborn`, `matplotlib` |
| Dashboard | `dash` (Plotly) |
| Environment | Google Colab (shared notebook) |

---

## Key Files

| File | What It Is |
|------|-----------|
| `CS 163 Project.ipynb` | Main Colab notebook — feature extraction + EDA code |
| `vocal_features.csv` | Extracted features (405 rows — belt/breathy only, will be expanded) |
| `EDA Summary - Google Docs.pdf` | Submitted EDA report |
| `Analysis_and_Visualization_Plan.md` | Submitted analysis plan |
| `Refined_Project_Proposal.md` | Updated proposal with full project scope |
| `geminiconvo.md` | Project context doc (tech stack, plan, AI agent instructions) |
| `EDA_WORK.md` | EDA code reference with explanations |
| `WORKFLOW_REPORT.md` | Status tracking doc |

---

## ML Models We're Using (and Why)

| Model | Why We Chose It | What We Get From It |
|-------|----------------|-------------------|
| **PCA** | Centroid and Rolloff are 89% correlated — causes multicollinearity | Reduces 4 features → 2 components; 2D scatter for visualization and dashboard |
| **Logistic Regression** | Need a simple baseline to benchmark against | Interpretable coefficients; quantifies how much non-linear models improve |
| **Random Forest** | Handles non-linear boundaries; robust to HNR outliers | Feature importance scores — validates that HNR is the #1 predictor |
| **XGBoost** | State-of-the-art on tabular data | Highest accuracy + SHAP values for per-sample interpretability |
| **Music2Emo** | Vocal technique can't distinguish positive from negative emotion | Valence scores that augment our arousal → full circumplex coverage |

---

## The Three-Stage Validation Story

This is the scientific backbone of the project:

```
Stage 1: VocalSet (CAN the model classify vocal energy?)
  Train on ~2,550 labeled recordings
  80/20 split → hard accuracy number
  "Yes, it distinguishes high-energy from low-energy techniques"

Stage 2: RAVDESS (DOES technique predict emotion?)
  Run SAME model on emotional singing (no retraining)
  angry/happy → should predict high energy
  calm/sad → should predict low energy
  "Yes, a technique-trained model correctly predicts emotional arousal"

Stage 3: FMA + Spleeter (DOES it work on real music?)
  Isolate vocals from real songs
  Run model → check against genre expectations
  "It generalizes to real-world music with X% accuracy gap"
```

Each stage answers a harder question. Together they prove the full thesis.

---

## Submitted Assignments & Grades

| Assignment | Score | Notes |
|-----------|-------|-------|
| Proposal Draft | 1 pt | Completion grade |
| Finalized Proposal | 6.5/11 | Lost points: vague objective (1.5/4), generic broader impacts (2/4), used "valence" when we measure "arousal" |
| EDA Summary | 13 pts | Submitted, graded |
| Analysis & Viz Plan | 10 pts | Submitted Mar 22 |

### Lessons Learned (Important)
1. **Numbers in documents MUST match the notebook.** The EDA report said "500 entries → 475 after cleaning" but the notebook has 405. This kind of inconsistency costs points.
2. **Be specific, not generic.** Don't write "Random Forest is good for classification." Write "Random Forest handles the non-linear class overlap and is robust to our HNR range of −86 to +24 dB."
3. **Every method choice needs a reason tied to YOUR data.** The grader wants to see that you picked methods because of what you found in EDA, not because a textbook listed them.
4. **Use correct terminology.** We measure arousal (energy level), NOT valence (positive/negative). The professor caught this mistake.
5. **Justify your core assumption.** Don't just claim vocal technique = emotion. Prove it with RAVDESS validation.

---

## How to Get Set Up

1. **Google Drive:** Get access to the shared drive `CS 163 Project/` — this has the dataset, CSV, and saved plots
2. **Colab Notebook:** Open `CS 163 Project.ipynb` from the shared drive
3. **Run cells 1–11** to load data and see the extracted features (you can skip cell 9's extraction since `vocal_features.csv` already exists — start from cell 13 which loads the CSV directly)
4. **EDA cells (13–24)** produce all the visualizations from the EDA Summary

### Important: Drive Path
The correct path is:
```
/content/drive/Shareddrives/CS 163 Project/
```
NOT `/content/drive/MyDrive/163 Project/` (this was an old path that caused issues).

---

## What's Next

### Immediate Priority
1. **Expand feature extraction** — Re-run extraction on all 10 techniques (~2,550 files instead of 405)
2. **Train models** — PCA, LogReg, RF, XGBoost on expanded dataset
3. **Evaluate** — ROC curves, confusion matrices, feature importance

### Then
4. **RAVDESS validation** — Download, extract features, run model, compute accuracy
5. **FMA + Spleeter** — Download FMA subset, isolate vocals, run model
6. **Music2Emo integration** — Add valence scores to FMA results
7. **Dashboard** — Build Dash web app
8. **Final Report & Presentation**

### Who Does What (Suggested)

| Person | Owns |
|--------|------|
| **Person 1** | Expand VocalSet extraction + model training |
| **Person 2** | RAVDESS download + validation pipeline |
| **Person 3** | FMA + Spleeter + Music2Emo integration |
| **New member** | Dashboard + report writing (use this doc to catch up) |

---

## Quick Reference: Key Numbers (from EDA — belt/breathy only)

| Metric | Value |
|--------|-------|
| Total samples (current) | 405 |
| Belt (high energy) | 205 |
| Breathy (low energy) | 200 |
| Clean samples (HNR ≥ −20) | 349 |
| Features | 4 (f₀, centroid, rolloff, HNR) |
| Centroid/Rolloff correlation | r = 0.89 |
| Belt mean HNR | −5.93 dB |
| Breathy mean HNR | −8.31 dB |
| HNR range (raw) | −86.2 to +23.8 dB |

**Note:** These numbers are from the original belt/breathy subset. Once we expand to 10 techniques (~2,550 samples), new EDA stats will need to be computed. Use these for consistency with already-submitted documents only.
