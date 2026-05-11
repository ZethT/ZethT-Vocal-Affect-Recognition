<a name="readme-top"></a>
<br />
<div align="center">
  <h3 align="center">Vocal-Affect-Recognition</h3>

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
* **High Energy:** Includes techniques like **Belt** and **Vibrato**.
* **Low Energy:** Includes techniques like **Breathy**, **Straight**, and **Vocal Fry**.

By leveraging **Mel-Frequency Cepstral Coefficients (MFCCs)** alongside traditional spectral features (F0, Spectral Centroid, Spectral Rolloff, HNR), the model successfully captures the timbral nuances that distinguish forceful singing from breathy or fry-based vocalizations.

## Directory Information

```text
.
├── vocal_features.csv
├── README.md
```

## Getting Started

Here we will describe the necessary actions and steps that should be followed in order to run this pipeline.

### Prerequisites

You will need Python 3.12+ and the following packages:
```sh
pip install librosa pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone [https://github.com/yourusername/vocalset-classification.git](https://github.com/yourusername/vocalset-classification.git)
   ```
2. Ensure the VocalSet audio data is present in the `/content/FULL/` directory or update the paths in `CS_163_Project.ipynb`.
3. Run the notebook to process the audio and generate `vocal_features.csv`.

## Usage
The pipeline follows a strict machine learning workflow:
* **Extraction:** `librosa` computes the Fundamental Frequency (F0), Spectral Centroid, Rolloff, HNR, and 13 MFCCs for each audio clip.
* **Preprocessing:** Data is standardized using a `StandardScaler` within a pipeline to handle the varying scales of frequency vs. coefficient data.
* **Training:** Multiple models (LogReg, Random Forest, XGBoost) are compared using an 80/20 train-test split.
* **Validation:** Performance is visualized via confusion matrices and ROC curves to ensure high recall for both energy classes.

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
