import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

# Note: This script contains the experimental approaches that were tested 
# but ultimately discarded due to lower performance (Domain Shift issues).
# They are preserved here for research documentation.

def bazin_function(time, A, B, t0, tau_fall, tau_rise):
    """
    The Bazin function (2009) used to model astronomical transients.
    Shape: A 'shark-fin' with exponential rise and fall.
    """
    exponent = -(time - t0) / tau_fall
    denominator = 1 + np.exp((time - t0) / tau_rise)
    return A * (np.exp(exponent) / denominator) + B

def fit_bazin(lightcurve_df):
    """
    Attempt 1: Fitting Bazin shapes to raw lightcurves.
    Hypothesis: TDEs will have a low residuals (chi-squared) compared to noise.
    Result: Failed on faint Test objects due to low Signal-to-Noise Ratio.
    """
    # Placeholder for optimization logic
    # In practice, this was run per-object, per-band
    pass

def run_isolation_forest(X_train, X_test):
    """
    Attempt 3: Unsupervised Anomaly Detection.
    Hypothesis: TDEs are 'anomalies' compared to standard stars.
    """
    print("Running Isolation Forest...")
    # Contamination set to approx TDE rate (148 / total)
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_train)
    
    # Predict (-1 is anomaly, 1 is normal)
    train_anom = iso.predict(X_train)
    test_anom = iso.predict(X_test)
    
    # Invert so 1 is the 'Anomaly' (Target)
    return np.where(test_anom == -1, 1, 0)

def apply_hard_physics_cut(df_predictions, redshift_col, color_col):
    """
    Attempt 7: The 'Hard Physics' Cut.
    Hypothesis: Enforce strict Cosmological rules (Z > 0, Color < 0.15).
    Result: Failed because metadata (Redshift) often contained '0.0' for missing values,
    causing the filter to delete valid TDEs.
    """
    print("Applying Hard Physics Filters...")
    
    # Rule 1: Must be extragalactic (Redshift > 0.01)
    is_galaxy = redshift_col > 0.01
    
    # Rule 2: Must be Hot/Blue (g-r Color < 0.15)
    is_hot = color_col < 0.15
    
    # Combine with ML probability (e.g., prob > 0.5)
    final_pred = df_predictions & is_galaxy & is_hot
    return final_pred.astype(int)

if __name__ == "__main__":
    print("This module contains archived experimental functions.")
    print("Import these into your main pipeline to reproduce failed experiments.")
