<a name="readme-top"></a>
<br />
<div align="center">
  <h3 align="center">Vocal-Affect-Recognition</h3>

  <p align="center">
    An automated machine learning pipeline for classifying vocal audio excerpts into high and low energy categories using Digital Signal Processing (DSP) and ensemble modeling.
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
