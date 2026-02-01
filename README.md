This repository contains the code and experimental history for my submission to the 2026 Kaggle MALLORN Challenge.

# Case Study: Hunting Black Holes with Machine Learning
## Lessons from the Kaggle MALLORN Challenge

**Project:** MALLORN Astronomical Classification Challenge
**Result:** Top 15% (Global Leaderboard)
**Score:** 0.1165 (F1-Score)

### Executive Summary
This repository documents the research, feature engineering, and experimental pipelines developed for the MALLORN challenge. The goal was to classify Tidal Disruption Events (TDEs)—rare astrophysical phenomena where a black hole destroys a star—using time-series photometric data.

Key achievements include:
1. Handling extreme class imbalance (less than 1% positive cases).
2. Diagnosing severe Covariate Shift between Training and Test data.
3. Developing a physics-informed feature engine that outperformed standard deep learning approaches on this dataset.

### The Data Challenge
The dataset consisted of light curves in six wavelengths (u, g, r, i, z, y).
* **The Signal:** 148 confirmed TDEs.
* **The Noise:** Thousands of imposters (AGNs, Variable Stars, Supernovae).
* **The Problem:** Traditional classifiers (Random Forest) achieved 99% accuracy by predicting "0" for everything, resulting in a failed F1-Score.

### Phase 1: The Baseline (Physics-Informed Features)
To overcome the noise, I engineered features based on astrophysical first principles rather than raw data statistics.

* **Extinction Correction:** Applied the Fitzpatrick (1999) reddening law to correct flux for interstellar dust absorption.
* **Blackbody Temperature:** Calculated color differentials (g-r, u-g) to isolate ultra-hot events (30,000K).
* **Burst Dynamics:** Calculated the ratio of Peak Flux to Mean Flux to quantify the "explosiveness" of the light curve.

**Result:** This physics-based approach yielded a baseline HistGradientBoosting score of 0.1165.

### Phase 2: Experimental Archive (Failed Attempts)
A significant portion of this project involved testing advanced hypotheses to break the 0.1165 ceiling. Ten distinct pivots were attempted. These experiments failed to improve the score but provided critical data on the nature of the dataset (Domain Shift).

1. **Bazin Parametric Fitting:** Attempted to fit a "shark-fin" mathematical function to light curves. Failed on faint test objects due to low signal-to-noise ratio.
2. **Contextual Metadata Filter:** Used Target Encoding on host galaxy metadata. Failed because Kaggle metadata contained missing labels.
3. **Isolation Forest (Anomaly Detection):** Attempted unsupervised detection. Failed because "anomalies" in astronomy are often just data glitches, not TDEs.
4. **Pseudo-Labeling:** Attempted Semi-Supervised learning. Failed by amplifying false positive confidence.
5. **Precision Thresholding:** Setting a 90% confidence threshold resulted in a 0.000 score, proving the model was never highly confident in the noisy test set.
6. **Automated Threshold Tuning:** Optimization on the training set did not transfer to the test set.
7. **Hard Physics Constraints:** Filtering by Redshift > 0.01 failed because missing values were coded as 0.0.
8. **Optuna Hyperparameter Tuning:** Resulted in overfitting to the training distribution.
9. **CatBoost (Ordered Boosting):** Failed to generalize despite overfitting protection.
10. **SMOTE (Synthetic Data):** Provided a minor improvement but did not beat the baseline.

### Phase 3: The Diagnosis
The collective failure of 10 advanced techniques pointed to a single root cause: **Covariate Shift**.

The Training Set consisted primarily of "Bright/Easy" TDEs, while the Test Set consisted of "Faint/Hard" TDEs. Models trained on absolute flux values learned to ignore faint objects.

### Repository Structure
* **src/physics_features.py**: The core feature engineering pipeline implementing Extinction and Color corrections.
* **experiments/01_baseline_model.py**: The successful code producing the 0.1165 score.
* **experiments/02_optuna_tuning.py**: The automated hyperparameter optimization script.
* **experiments/03_smote_ensemble.py**: Implementation of Synthetic Minority Over-sampling and Voting Classifiers.
* **experiments/04_experimental_archive.py**: Archived logic for Bazin fits, Isolation Forests, and Physics filters (for documentation purposes).

### Tools Used
* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, CatBoost, Optuna, Imbalanced-learn, Extinction, SciPy
