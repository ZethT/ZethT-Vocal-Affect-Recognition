<a name="readme-top"></a>
<br />
<div align="center">
  <h3 align="center">Vocal-Affect-Recognition</h3>

  <p align="center">
    An automated machine learning pipeline for classifying vocal audio excerpts into high and low energy categories using Digital Signal Processing (DSP) and ensemble modeling.
    <br />
    <br />
    <a href="https://github.com/yourusername/vocalset-classification/issues">Report Bug</a>
    ·
    <a href="https://github.com/yourusername/vocalset-classification/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#directory-information">Directory Information</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

This project addresses the challenge of automatically identifying the "energy" of a vocal performance using machine learning. By analyzing the **VocalSet** dataset—a collection of professional singer recordings—the pipeline transforms raw audio into a feature-rich dataset to distinguish between high-intensity and low-intensity vocalizations.

**The Classification Logic:**
* **High Energy:** Includes techniques like **Belt** and **Vibrato**.
* **Low Energy:** Includes techniques like **Breathy**, **Straight**, and **Vocal Fry**.

By leveraging **Mel-Frequency Cepstral Coefficients (MFCCs)** alongside traditional spectral features (F0, Spectral Centroid, Spectral Rolloff, HNR), the model successfully captures the timbral nuances that distinguish forceful singing from breathy or fry-based vocalizations.

## Directory Information

```text
.
├── CS_163_Project.ipynb      # Main development notebook (Data processing, EDA, Training)
├── vocal_features.csv        # Processed dataset with 17 extracted audio features
├── feature_names.joblib      # Serialized list of feature names for model consistency
├── vocal_model_lr.joblib     # Trained Logistic Regression pipeline (Best Model)
├── vocal_model_rf.joblib     # Trained Random Forest classifier
├── README.md                 # Project documentation
├── cs163-final-rubric.pdf    # Project grading criteria
└── images/                   # Visualizations generated during training
    ├── image_285215.png      # Confusion Matrix
    ├── image_28007e.png      # Feature Importance/Correlation plot
    ├── image_28003f.png      # ROC Curve
    ├── image_27fc3c.png      # Spectral Analysis plot
    ├── image_27fc22.png      # Feature Distribution
    └── image_27fc05.png      # Model comparison chart
