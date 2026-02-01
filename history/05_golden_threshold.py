import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ==========================================
# EXPERIMENT: AUTOMATED THRESHOLD OPTIMIZATION
# Hypothesis: We can mathematically find the perfect probability cutoff
#             by maximizing the F1-Score on Cross-Validation data.
# ==========================================

# 1. Load Data
train_lc = pd.read_csv('../data/train_lightcurves.csv')
train_meta = pd.read_csv('../data/train_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')
test_meta = pd.read_csv('../data/test_log.csv')

def features(df):
    return df.groupby('object_id')['Flux'].agg(['mean', 'std', 'skew']).reset_index()

train_feat = features(train_lc).merge(train_meta[['object_id', 'target']], on='object_id')
test_feat = features(test_lc).merge(test_meta[['object_id']], on='object_id')

X = train_feat.drop(columns=['object_id', 'target'])
y = train_feat['target']
X_test = test_feat.drop(columns=['object_id'])

# 2. Cross-Validation to find Best Threshold
print("Running Cross-Validation to find Golden Threshold...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_thresholds = []
val_scores = []

for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_tr, y_tr)
    
    # Predict Probabilities
    val_probs = model.predict_proba(X_val)[:, 1]
    
    # Search for Best Threshold (0.01 to 0.99)
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    best_thresh = 0.5
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    best_thresholds.append(best_thresh)
    val_scores.append(best_f1)

# Average the best thresholds found in each fold
golden_threshold = np.mean(best_thresholds)
print(f"Optimal Threshold Found: {golden_threshold:.4f} (Avg Validation F1: {np.mean(val_scores):.4f})")

# 3. Apply to Test Set
print("Applying Golden Threshold to Test Data...")
final_model = XGBClassifier(random_state=42)
final_model.fit(X, y)
test_probs = final_model.predict_proba(X_test)[:, 1]

# Use the exact number found during training
final_preds = (test_probs >= golden_threshold).astype(int)

sub = pd.DataFrame({'object_id': test_meta['object_id'], 'prediction': final_preds})
sub.to_csv('submission_golden_cut.csv', index=False)
print("Done.")
