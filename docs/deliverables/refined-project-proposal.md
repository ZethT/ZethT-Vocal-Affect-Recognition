Tanzil Ahmed, James Kim, Zeth Tang
CS 163 Sec 01
22 March 2026

# Project Proposal: Vocal Affect Recognition
## Enhancing Music Recommendation via Timbral Mood Classification

**Group ID:** 16

---

## 1. Project Summary

Current music recommendation systems classify songs by metadata — genre tags, artist names, and collaborative filtering from user listening history. These methods fail to capture the emotional content delivered through the singer's voice. A "Chill" playlist might include a high-energy belted power ballad simply because other chill-playlist listeners also played it. The algorithm never actually listened to the voice.

This project builds a **Mood Tagger** that classifies a song's emotional energy by analyzing the **acoustic properties of the singing voice**. Rather than treating mood as a subjective tag, we measure it through the physics of vocal production:

- **High-energy vocal delivery** (belt, forte, vibrato) produces strong harmonic structure, high spectral brightness, and a high Harmonic-to-Noise Ratio (HNR) — the vocal folds close completely with each vibration.
- **Low-energy vocal delivery** (breathy, pianissimo, vocal fry) produces turbulent airflow, diffuse spectral energy, and low HNR — the vocal folds do not fully close.

These are measurable physical properties, not subjective opinions.

**How it works:**

1. **Train** an arousal classifier on ~2,550 labeled vocal recordings from the VocalSet dataset, using 10 vocal techniques grouped into high-energy and low-energy categories. We extract four acoustic features: fundamental frequency (f₀), spectral centroid, spectral rolloff, and harmonic-to-noise ratio (HNR).
2. **Validate the technique-emotion link** by running the trained model (without retraining) on RAVDESS emotional singing recordings. If a model trained only on vocal technique correctly predicts emotional arousal (angry/happy = high energy, calm/sad = low energy), this proves that vocal technique is a valid proxy for emotion.
3. **Validate on real music** by using Spleeter to isolate vocal stems from Free Music Archive (FMA) tracks, extracting the same features, and checking predictions against genre expectations.
4. **Augment with valence** using Music2Emo, a pre-trained emotion recognition model, to add the positive/negative dimension that vocal technique alone cannot capture. This enables full Russell's Circumplex coverage.
5. **Deploy** as a web dashboard where a user uploads any song, the system isolates the vocal track, extracts features, and outputs a mood classification with an interpretable explanation of why.

**What makes this different from existing tools:** Spotify and similar platforms analyze the full audio mix with black-box deep learning models. They can tell you a song's energy score is 0.8 but cannot explain why. Our system analyzes the voice specifically, and explains its prediction through measurable acoustic properties: *"This song is high-energy because the vocal HNR is −4 dB with a spectral centroid of 3,500 Hz, indicating belt-like technique with strong harmonic closure."* This interpretability is our core contribution.

---

## 2. Broader Impacts

### Music Information Retrieval (MIR)
This project directly addresses the **Cold Start problem** in streaming platforms. When a new song is uploaded, it has zero listening history — collaborative filtering cannot recommend it. Our system can tag it instantly based on vocal analysis alone, enabling placement into mood-based playlists from the moment of upload. The voice-specific approach also solves the **instrumental contamination problem** — where loud drums make a softly sung track score "high energy" in full-mix analysis.

### Vocal Pedagogy
Singing instructors assess technique changes (chest voice, head voice, breathy) by ear — a subjective process that varies between teachers. Our feature extraction pipeline provides an **objective, quantitative feedback loop**. A student can record themselves, extract HNR and spectral features, and see numerically whether their delivery shifted toward a more powerful or more breathy technique. This makes technique assessment reproducible and measurable.

### Speech Pathology
Breathiness is a clinical marker for vocal pathology — conditions like vocal nodules, polyps, and vocal fold paralysis all reduce HNR by preventing complete glottal closure. While our system is not a diagnostic tool, the HNR extraction pipeline could be adapted for **screening applications** where a sustained drop in HNR over time flags potential vocal health issues for clinical evaluation.

### Sonic Branding & Media Production
Advertisers select music for campaigns based on emotional fit — a car commercial wants "powerful and confident" while a skincare ad wants "soft and intimate." Currently this is done by manual listening across large audio libraries. Our classifier enables **searchable audio libraries** filtered by measurable vocal energy level, replacing subjective listening with objective acoustic criteria.

---

## 3. Data Sources

1. **VocalSet:** Wilkins, J., Seetharaman, P., Wahl, A., & Pardo, B. (2018). *VocalSet: A Singing Voice Dataset for the Deep Learning Age.* [https://zenodo.org/record/1193957](https://zenodo.org/record/1193957)
   - 20 professional singers, 18 vocal techniques, 3,613 total recordings
   - We use 10 techniques grouped into high-energy (~1,350 samples: belt, forte, fast_forte, slow_forte, vibrato) and low-energy (~1,200 samples: breathy, pp, slow_piano, vocal_fry, spoken)

2. **RAVDESS:** Livingstone, S. R. & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)
   - 24 professional actors singing in 5 emotions (calm, happy, sad, angry, fearful) at 2 intensity levels
   - Used for cross-domain validation: proving vocal technique correlates with emotional arousal

3. **Free Music Archive (FMA):** Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). *FMA: A Dataset for Music Analysis.* [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
   - Real-world songs with genre metadata
   - Used for real-world validation via Spleeter vocal isolation

4. **Spleeter:** Hennequin, R., Khlif, A., Felix, A., & Monte, M. (2020). *Spleeter: A Fast and Efficient Source Separation Library.* Deezer Research.
   - Pre-trained neural network for vocal/accompaniment separation

5. **Music2Emo:** AMAAI Lab (2025). *Music2Emo: Towards Unified Music Emotion Recognition.* [https://github.com/AMAAI-Lab/Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion)
   - Pre-trained model for valence and arousal prediction
   - Used to augment our arousal classification with valence for full circumplex coverage

6. **Russell's Circumplex Model of Affect:** Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178.
   - Theoretical framework: emotions mapped onto arousal (energy) and valence (positivity) axes
   - Our system addresses the arousal dimension through vocal analysis, augmented with Music2Emo for valence

---

## 4. Expected Major Findings / Project Questions

**Hypothesis 1:** High-energy vocal techniques (belt, forte, vibrato) will exhibit significantly higher HNR and Spectral Centroid than low-energy techniques (breathy, pp, vocal_fry), and this separation will hold across all 10 techniques — not just the belt/breathy extremes.

**Hypothesis 2:** A classifier trained exclusively on vocal technique labels (VocalSet) will correctly predict emotional arousal in RAVDESS singing — angry/happy singing classified as high-energy, calm/sad singing as low-energy — without ever seeing emotion labels during training. This would demonstrate that vocal technique is a measurable proxy for emotional arousal.

**Hypothesis 3:** Spleeter vocal isolation will introduce artifacts that degrade HNR in extracted features, causing accuracy to drop on FMA compared to VocalSet and RAVDESS. Quantifying this gap informs whether the pipeline requires recalibration for real-world deployment.

**Expected Outcome:** An interpretable arousal classifier that (1) achieves strong accuracy on VocalSet technique classification, (2) demonstrates cross-domain transfer to emotional arousal on RAVDESS, and (3) produces genre-consistent predictions on real-world FMA tracks. Combined with Music2Emo's valence, the system achieves full Russell's Circumplex mood classification — with the arousal dimension explainable through vocal acoustics.

---

## 5. Analysis Plan

### Training Data
~2,550 VocalSet recordings across 10 techniques, grouped into:
- **High energy:** belt (205), forte (100), fast_forte (394), slow_forte (395), vibrato (255)
- **Low energy:** breathy (200), pp (100), slow_piano (397), vocal_fry (198), spoken (20)

### Models

| Model | Purpose |
|-------|---------|
| PCA | Reduce 4 features to 2 principal components; handle centroid/rolloff multicollinearity |
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Non-linear classification + Gini feature importance |
| XGBoost | Highest accuracy + SHAP interpretability |

### Three-Stage Validation

| Stage | Data | What It Proves |
|-------|------|---------------|
| 1. Technique classification | VocalSet 80/20 split | Model can distinguish vocal energy levels |
| 2. Emotion transfer | RAVDESS (no retraining) | Vocal technique is a valid proxy for emotional arousal |
| 3. Real-world generalization | FMA + Spleeter | System works on commercial music |

### Valence Augmentation
Music2Emo provides valence scores for FMA tracks, combined with our arousal predictions for full circumplex mapping.

### Visualizations
- PCA 2D scatter plot (cluster separation across 10 techniques)
- ROC curves (multi-model comparison)
- Confusion matrices (per-model error profiles)
- Feature importance bar chart (RF Gini + XGBoost SHAP)
- Three-stage accuracy comparison (VocalSet → RAVDESS → FMA)
- Russell's Circumplex scatter plot (arousal × valence for FMA tracks, colored by genre)
