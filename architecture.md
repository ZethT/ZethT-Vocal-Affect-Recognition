# Cursor Prompt — Build CS163 Vocal Affect Recognition Final Project Website

You are helping me build the final submission for my CS163 Data Science Senior Project.

The project is:

# Vocal Affect Recognition:
## Enhancing Music Recommendation via Timbral Mood Classification

The project uses machine learning and audio signal processing to classify vocal affect based on singing technique using the VocalSet dataset.

We classify:
- `belt` vocals → high energy / high arousal
- `breathy` vocals → low energy / intimate / low arousal

The ML side is already mostly complete:
- audio preprocessing
- feature extraction
- EDA
- model training
- Random Forest classifier
- confusion matrix
- feature importance
- PCA
- ROC curves
- MFCC analysis

Now we need to build the FINAL FULL-STACK DEPLOYED PROJECT that satisfies the course rubric.

---

# IMPORTANT IMPLEMENTATION DECISION

DO NOT use React, Next.js, TypeScript, Redux, or any complicated frontend framework.

Use a simpler stack:

# Frontend Stack
- HTML
- CSS
- Vanilla JavaScript

# Backend Stack
- Python
- FastAPI
- scikit-learn
- librosa
- pandas
- numpy
- joblib
- praat-parselmouth

# Cloud / Deployment
- Google App Engine for frontend hosting
- Google Cloud Run for ML inference backend
- Google Firestore for cloud database storage

This is an academic project website, NOT a production SaaS app.

Prioritize:
- simplicity
- readability
- clean design
- deployment reliability
- rubric completion

Avoid overengineering.

---

# Project Requirements From Rubric

The final website must include:

1. Landing page with informative visualizations and concise explanation
2. Project Objective page
3. Analytical Methods page
4. Major Findings page
5. At least one interactive diagram or interactive feature
6. Hosted on Google App Engine
7. At least part of the data stored in cloud database
8. ML inference service running on Cloud Run as Docker app

The GitHub repo must include:
- organized codebase
- setup instructions
- pipeline explanation
- system design
- inference service explanation
- cloud storage explanation
- reasonable commit structure
- deployment instructions

---

# Full Architecture

```text
User
 |
 | opens website
 v
Frontend Website (HTML/CSS/JS)
Hosted on Google App Engine
 |
 | upload .wav file using fetch()
 v
FastAPI ML Inference Service
Hosted on Google Cloud Run
 |
 | extract audio features
 | run Random Forest model
 v
Prediction JSON returned
 |
 v
Frontend displays:
- predicted class
- confidence
- probabilities
- extracted features
 |
 v
Prediction metadata stored in Firestore