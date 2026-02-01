This repository contains the code and experimental history for my submission to the 2026 Kaggle MALLORN Challenge.

Case Study: Hunting Black Holes with Machine Learning – Lessons from the MALLORN Kaggle Challenge
Executive Summary
Project: Kaggle MALLORN Challenge (Classifying Tidal Disruption Events in astronomical time-series data).

Final Rank: Top 15% (~120th out of ~800 global teams).

Final Score: 0.1165 (F1-Score).

Key Insight: Standard Machine Learning fails under extreme Domain Shift; Physics-informed feature engineering is the ultimate differentiator.

Introduction
In the intersection of Big Data and Astrophysics lies one of the most exciting challenges in modern science: detecting the invisible. As we prepare for next-generation telescopes like the Vera C. Rubin Observatory, we will soon be flooded with millions of astronomical alerts every night.

In this project, I tackled the Kaggle MALLORN Challenge: Can we use Machine Learning to identify Tidal Disruption Events (TDEs) from raw light curve data?

A TDE occurs when a supermassive black hole tears apart a wandering star. They are incredibly rare, scientifically invaluable, and buried in a sea of data noise. With my dual background in Physics and Data Analytics, I built a custom classification engine to find these cosmic needles in a haystack, ultimately placing in the Top 15% of a global leaderboard.

The Data Challenge: Extreme Imbalance & High Noise
The dataset consisted of time-series photometric data (light curves) in various wavelengths (u, g, r, i, z, y).

The Signal: Only 148 confirmed TDEs in the training data.

The Noise: Thousands of "Imposters" like Active Galactic Nuclei (AGNs), Variable Stars, and Supernovae that mimic TDE behavior.

Phase 1: Setting the Baseline (Physics-Informed Features)
Traditional algorithms fail immediately on this data. A standard Random Forest would simply predict "Not a TDE" for everything to achieve 99% accuracy. Before modeling, I translated the raw numbers into physical properties:

Extinction Correction (De-dusting): Using the Fitzpatrick (1999) law, I corrected the flux values for interstellar dust.

Color Temperature (Blackbody Radiation): TDEs are 30,000K fireballs. I calculated color differentials (e.g., g - r) to isolate ultra-hot events.

Burst Dynamics: TDEs have a fast rise and slow decay. I calculated Peak-to-Mean flux ratios.

Using a Histogram-based Gradient Boosting Classifier, this engine achieved a benchmark F1-Score of 0.1165. This score became the foundation of my success, but it hit a hard ceiling. To break it, I initiated an exhaustive series of 10 advanced data science experiments.

Phase 2: The Experimental Journey (10 Pivots)
The following is the complete technical breakdown of the techniques I tested, the hypothesis behind them, and how their failures informed the next iteration.

Attempt 1: The "Pure Physics" Shape Matcher (Bazin Fit)

The Hypothesis: Standard trees only look at data points, not the shape of time-series data. I used scipy.optimize to mathematically force a "shark-fin" curve (Bazin Function) onto the data.

The Pivot: The Bazin fit worked beautifully on bright TDEs but failed completely on faint ones, where the noise was too high to fit a curve. (Score: 0.1123).

Attempt 2: The Contextual Filter (Host Galaxy Metadata)

The Hypothesis: TDEs cannot happen on isolated stars. I used Target Encoding on the SpecType column to filter out impossible environments (like M-Dwarf stars).

The Pivot: The Kaggle metadata was incomplete. By filtering for known galaxies, the model accidentally wiped out hidden TDEs labeled as "Unknown". (Score: 0.0959).

Attempt 3: Unsupervised Anomaly Detection (Isolation Forest)

The Hypothesis: With only 148 targets, I used an Isolation Forest to map the "Normal" universe (AGNs/Stars) and flagged anything that broke the rules as an Anomaly.

The Pivot: In astronomy, 99% of "anomalies" are just broken telescope pixels or data glitches. (Score: 0.0959).

Attempt 4: Semi-Supervised Learning (Pseudo-Labeling)

The Hypothesis: I ran the baseline model on the Test Set, identified predictions with >95% confidence, and appended them to the training data as "Pseudo-labels" to increase training size.

The Pivot: The model became too confident in its own errors. False positives were amplified. (Score: 0.0976).

Attempt 5: The "Sniper" Precision Cut

The Hypothesis: To reduce False Positives, I set a strict threshold: only classify as a TDE if the model is >90% confident.

The Pivot: Score: 0.000. The model was never 90% confident because the data was too noisy. It predicted zero TDEs.

Attempt 6: Automated Threshold Tuning (The Golden Cut)

The Hypothesis: I used an automated script to test 100 different thresholds to find the exact mathematical peak of the F1-Score on the training set.

The Pivot: The optimal threshold on the Training Set performed terribly on the Test Set, proving the two datasets were different. (Score: 0.0707).

Attempt 7: The "Hard Physics Cut" (No ML)

The Hypothesis: I abandoned ML probabilities. I hard-coded Cosmological rules: Redshift > 0.01 (Must be a galaxy) and Color g-r < 0.15 (Must be hot).

The Pivot: Score: 0.0110. This taught a crucial lesson: In simulations, metadata is often garbage. Missing redshifts were coded as '0', so my physics filter accidentally deleted half the real TDEs.

Attempt 8: AI-Driven Tuning (Optuna + XGBoost GPU)

The Hypothesis: I utilized Optuna with GPU-accelerated XGBoost to test 30+ model configurations (depth, learning rate, regularization) automatically.

The Pivot: The optimizer achieved near-perfect scores on the Training Set but failed on the Test Set (Overfitting). (Score: 0.0971).

Attempt 9: The Overfit Defense (CatBoost)

The Hypothesis: To stop overfitting, I switched to CatBoost, utilizing its "Ordered Boosting" algorithm designed specifically to prevent data leakage in small datasets.

The Pivot: Even the world's best anti-overfitting model failed to raise the score, confirming the issue was the data, not the algorithm. (Score: 0.0969).

Attempt 10: Synthetic Cloning (SMOTE)

The Hypothesis: 148 TDEs are too few to learn from. I used SMOTE (Synthetic Minority Over-sampling Technique) to artificially clone and interpolate new synthetic TDEs in the vector space.

The Pivot: It provided the highest score of the advanced experiments, but still fell short of the baseline. (Score: 0.0979).

Final Attempt: The Trifecta Ensemble (Soft Voting HGB + RF + ExtraTrees).

Phase 3: The Grand Diagnosis – "Domain Shift"
By analyzing the collective failure of these 10 advanced pipelines, I uncovered the central trap of the competition. The dataset suffered from severe Covariate Shift (Domain Shift).

The Training Set contained "Easy" TDEs—incredibly bright and obvious.

The Test Set contained "Hidden" TDEs—faint, distant, and noisy.

Every model I trained learned the same false rule: "If it is bright, it is a TDE." Consequently, when the models looked at the faint Test Set, they dismissed the actual targets as noise.

To prove this, I ran a final "Scale-Invariant" experiment, deleting all absolute brightness metrics (mean, max flux) and forcing the model to look only at color and shape ratios. While it stabilized the predictions, the loss of absolute luminosity meant the hard ceiling of 0.1165 remained the absolute peak for standard ML.

Final Results & Takeaway
While 0.1165 may seem low in standard ML tasks, in the context of astronomical transient detection, it is highly competitive. Surpassing roughly 85% of competitors, this score was achieved not through heavy compute power, but through domain-specific feature engineering.

This project proved that domain knowledge (Astrophysics) is essential for diagnosing data that standard algorithms cannot comprehend. For future telescope surveys, we cannot rely on algorithmic tricks; we must combine statistical ML with multi-wavelength constraints (X-Ray + Optical) to bridge the gap between faint signal and background noise.

Tools & Techniques Explored: Python, Pandas, Scikit-Learn, XGBoost, CatBoost, Scipy, Optuna, SMOTE, Isolation Forest, Pseudo-labeling, Voting Classifiers, Extinction Correction.
