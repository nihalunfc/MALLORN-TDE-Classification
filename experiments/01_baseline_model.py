import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics_features import extract_features

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

# --- 1. Load Data ---
# Note: Users should update paths to their local data location
print("Loading Data...")
train_log = pd.read_csv('../data/train_log.csv')
train_lc = pd.read_csv('../data/train_lightcurves.csv') # Concatenated lightcurves
test_log = pd.read_csv('../data/test_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')

# --- 2. Feature Engineering ---
print("Extracting Physics Features...")
train_final = extract_features(train_lc, train_log)
test_final = extract_features(test_lc, test_log)

# --- 3. Modeling (HistGradientBoosting) ---
X = train_final.drop(columns=['object_id', 'target'])
y = train_final['target']
X_test = test_final.drop(columns=['object_id'])

# Simple Training Loop
model = HistGradientBoostingClassifier(
    learning_rate=0.04, 
    max_iter=300, 
    class_weight='balanced', 
    random_state=42
)

print("Training Model...")
model.fit(X, y)

# --- 4. Prediction ---
probs = model.predict_proba(X_test)[:, 1]

# Dynamic Thresholding (Top 10% Strategy)
base_rate = y.mean()
target_count = int(len(X_test) * base_rate * 2.0)
sorted_probs = np.sort(probs)
thresh = sorted_probs[-target_count]

final_preds = (probs >= thresh).astype(int)

# Output
submission = pd.DataFrame({'object_id': test_final['object_id'], 'prediction': final_preds})
submission.to_csv('submission_baseline.csv', index=False)
print(f"Submission generated. Threshold used: {thresh}")
