import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

# ==========================================
# EXPERIMENT: HIGH-PRECISION THRESHOLDING ("THE SNIPER")
# Hypothesis: Maximizing Precision by setting an extremely high 
#             probability threshold (>0.90) to eliminate False Positives.
# Result: FAILED (Score 0.0). The model was never confident enough.
# ==========================================

# 1. Load Data
print("Loading Data...")
train_lc = pd.read_csv('../data/train_lightcurves.csv')
train_meta = pd.read_csv('../data/train_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')
test_meta = pd.read_csv('../data/test_log.csv')

# Feature Extraction (Standard Mean/Max)
def quick_features(df):
    return df.groupby('object_id')['Flux'].agg(['mean', 'max', 'std']).reset_index()

train_feat = quick_features(train_lc).merge(train_meta[['object_id', 'target']], on='object_id')
test_feat = quick_features(test_lc).merge(test_meta[['object_id']], on='object_id')

X = train_feat.drop(columns=['object_id', 'target'])
y = train_feat['target']
X_test = test_feat.drop(columns=['object_id'])

# 2. Train Model
# We use a standard classifier, no class weights, to see 'raw' probabilities
print("Training Probability Model...")
model = XGBClassifier(
    n_estimators=300, 
    max_depth=5, 
    learning_rate=0.05,
    random_state=42
)
model.fit(X, y)

# 3. Predict Probabilities
probs = model.predict_proba(X_test)[:, 1]

# 4. The "Sniper" Logic
# Instead of a dynamic threshold (like Top 10%), we use a Hard Cutoff.
SNIPER_THRESHOLD = 0.90

print(f"Applying Hard Cutoff: {SNIPER_THRESHOLD}")
print(f"Max Probability found in Test Set: {probs.max():.4f}")

# This will result in all Zeros if max_prob < 0.90
preds = (probs >= SNIPER_THRESHOLD).astype(int)

if preds.sum() == 0:
    print("WARNING: Model predicted 0 events. Threshold is too high.")

# 5. Save
sub = pd.DataFrame({'object_id': test_meta['object_id'], 'prediction': preds})
sub.to_csv('submission_sniper.csv', index=False)
