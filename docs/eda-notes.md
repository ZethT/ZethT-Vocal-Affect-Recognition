# CS 163 — EDA Work Document
**Vocal Affect Recognition | Exploratory Data Analysis**
*Scope: Descriptive Statistics + Visualizations*

---

## 1. Dataset Summary

| Property | Value |
|----------|-------|
| Total samples | 405 |
| Belt samples | 205 |
| Breathy samples | 200 |
| Features extracted | 4 |
| File format | CSV (`vocal_features.csv`) |

### Features

| Column | Type | Description |
|--------|------|-------------|
| `file_path` | string | Path to source audio file |
| `technique` | string | Vocal technique label (`belt` / `breathy`) |
| `class` | int | Numeric class label (0 = breathy, 1 = belt) |
| `f0` | float | Fundamental frequency in Hz (mean per file) |
| `spectral_centroid` | float | Spectral centroid in Hz |
| `spectral_rolloff` | float | Spectral rolloff frequency in Hz |
| `hnr` | float | Harmonics-to-noise ratio in dB |

---

## 2. Data Quality Check

**Known issues:**
- HNR values range from -86.2 to 23.8 dB. Normal voiced speech is 10–25 dB. Values below -20 dB are likely Praat extraction artifacts (silent frames, unvoiced segments, or failed pitch detection).

### Code: Load Data + Quality Check

```python
import pandas as pd
import numpy as np

# Load from shared drive
df = pd.read_csv('/content/drive/Shareddrives/CS 163 Project/vocal_features.csv')

print("Shape:", df.shape)
print("\nClass distribution:")
print(df['technique'].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Flag HNR outliers
hnr_outliers = df[df['hnr'] < -20]
print(f"\nHNR outliers (hnr < -20 dB): {len(hnr_outliers)} rows")
print(hnr_outliers[['file_path', 'technique', 'hnr']])

# Create cleaned version (filter out HNR artifacts)
df_clean = df[df['hnr'] >= -20].copy()
print(f"\nClean dataset shape: {df_clean.shape}")
```

---

## 3. Descriptive Statistics

### Code: Full + Per-Class Stats

```python
features = ['f0', 'spectral_centroid', 'spectral_rolloff', 'hnr']

# Overall stats
print("=== OVERALL DESCRIPTIVE STATS ===")
print(df_clean[features].describe().round(2).to_string())

# Per-class stats
print("\n=== PER-CLASS STATS ===")
for technique in ['belt', 'breathy']:
    print(f"\n--- {technique.upper()} ---")
    subset = df_clean[df_clean['technique'] == technique][features]
    print(subset.describe().round(2).to_string())
```

### Expected Output Format (paste into PDF report)

| Stat | f0 (Hz) | spectral_centroid (Hz) | spectral_rolloff (Hz) | hnr (dB) |
|------|---------|----------------------|---------------------|----------|
| count | — | — | — | — |
| mean | — | — | — | — |
| std | — | — | — | — |
| min | — | — | — | — |
| 25% | — | — | — | — |
| 50% | — | — | — | — |
| 75% | — | — | — | — |
| max | — | — | — | — |

*(Run the code above to fill in values from the actual data.)*

---

## 4. Correlation Matrix

### Code: Compute + Print Correlation Matrix

```python
print("=== CORRELATION MATRIX ===")
corr_matrix = df_clean[features].corr().round(3)
print(corr_matrix.to_string())
```

### Interpretation Guide

- **f0 ↔ spectral_centroid**: Expected positive correlation — higher pitch tends to shift spectral energy upward
- **f0 ↔ hnr**: Belt singing (higher f0) is more periodic, so positive correlation expected
- **spectral_centroid ↔ spectral_rolloff**: Strong positive correlation expected — both track high-frequency energy
- **hnr ↔ spectral features**: Breathy voice has more noise (lower HNR), possibly lower spectral centroid — negative correlation possible

---

## 5. Visualizations

### 5a. Histograms — Feature Distributions by Class

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distributions by Vocal Technique', fontsize=14)

colors = {'belt': '#2196F3', 'breathy': '#FF9800'}

for ax, feature in zip(axes.flatten(), features):
    for technique, color in colors.items():
        subset = df_clean[df_clean['technique'] == technique][feature]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=technique)
    ax.set_title(feature)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig('/content/drive/Shareddrives/CS 163 Project/hist_features.png', dpi=150)
plt.show()
```

### 5b. Boxplots — Feature Spread by Class

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distributions (Boxplot) by Vocal Technique', fontsize=14)

for ax, feature in zip(axes.flatten(), features):
    sns.boxplot(data=df_clean, x='technique', y=feature, ax=ax,
                palette={'belt': '#2196F3', 'breathy': '#FF9800'})
    ax.set_title(feature)
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig('/content/drive/Shareddrives/CS 163 Project/boxplot_features.png', dpi=150)
plt.show()
```

### 5c. Scatter Plot — f0 vs Spectral Centroid

```python
fig, ax = plt.subplots(figsize=(8, 6))

for technique, color in colors.items():
    subset = df_clean[df_clean['technique'] == technique]
    ax.scatter(subset['f0'], subset['spectral_centroid'],
               alpha=0.6, color=color, label=technique, s=40)

ax.set_xlabel('Fundamental Frequency f0 (Hz)')
ax.set_ylabel('Spectral Centroid (Hz)')
ax.set_title('f0 vs Spectral Centroid by Vocal Technique')
ax.legend()

plt.tight_layout()
plt.savefig('/content/drive/Shareddrives/CS 163 Project/scatter_f0_centroid.png', dpi=150)
plt.show()
```

### 5d. Correlation Heatmap

```python
fig, ax = plt.subplots(figsize=(7, 6))

corr_matrix = df_clean[features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, ax=ax,
            xticklabels=['f0', 'centroid', 'rolloff', 'HNR'],
            yticklabels=['f0', 'centroid', 'rolloff', 'HNR'])
ax.set_title('Acoustic Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('/content/drive/Shareddrives/CS 163 Project/heatmap_corr.png', dpi=150)
plt.show()
```

---

## 6. Preliminary Insights

### Hypotheses to Validate

| Hypothesis | Feature(s) | Expected Pattern |
|------------|-----------|-----------------|
| Belt has higher pitch | f0 | Belt median f0 > Breathy median f0 |
| Belt has brighter timbre | spectral_centroid, spectral_rolloff | Belt values higher |
| Belt is more periodic | hnr | Belt HNR > Breathy HNR |
| Features are correlated | all | Centroid ↔ rolloff should be strongly correlated |

### Notes on HNR Outliers
If the cleaned dataset (`hnr >= -20`) drops a meaningful number of rows, report:
- How many rows were removed
- Whether removals are evenly distributed across classes (if one class has disproportionately more artifacts, note this as a data quality limitation)

---

## Rubric Coverage

| Rubric Item | Covered By |
|-------------|-----------|
| Descriptive statistics | Section 3 — overall + per-class stats table |
| Correlation analysis | Section 4 — matrix + interpretation |
| Visualizations | Section 5 — histograms, boxplots, scatter, heatmap |
| Data quality | Section 2 — missing values, duplicates, outlier flag |
| Class breakdown | Section 3 — separate stats for belt vs. breathy |
