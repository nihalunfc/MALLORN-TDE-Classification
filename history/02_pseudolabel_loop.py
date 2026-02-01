import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

# ==========================================
# EXPERIMENT: SEMI-SUPERVISED PSEUDO-LABELING
# Hypothesis: Augment small training data by adding high-confidence 
#             Test predictions as new "Pseudo-Ground Truth".
# ==========================================

# 1. Load Data
print("Loading Data...")
train_lc = pd.read_csv('../data/train_lightcurves.csv')
train_meta = pd.read_csv('../data/train_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')
test_meta = pd.read_csv('../data/test_log.csv')

# (Simplified Feature Extraction for the Archive)
def quick_features(df):
    return df.groupby('object_id')['Flux'].agg(['mean', 'max', 'std']).reset_index()

train_feat = quick_features(train_lc).merge(train_meta[['object_id', 'target']], on='object_id')
test_feat = quick_features(test_lc).merge(test_meta[['object_id']], on='object_id')

X = train_feat.drop(columns=['object_id', 'target'])
y = train_feat['target']
X_test = test_feat.drop(columns=['object_id'])

# 2. Train Initial Teacher Model
print("Phase 1: Training Teacher Model...")
model_teacher = XGBClassifier(n_estimators=200, max_depth=4, random_state=42)
model_teacher.fit(X, y)

# 3. Generate Pseudo-Labels
print("Phase 2: Generating Pseudo-Labels...")
test_probs = model_teacher.predict_proba(X_test)[:, 1]

# CRITICAL STEP: Select only High Confidence predictions
# If prob > 0.95, we assume it's a TDE.
# If prob < 0.05, we assume it's NOT a TDE.
high_conf_indices = np.where((test_probs > 0.95) | (test_probs < 0.05))[0]
pseudo_X = X_test.iloc[high_conf_indices]
pseudo_y = (test_probs[high_conf_indices] > 0.5).astype(int)

print(f"Added {len(pseudo_X)} new pseudo-samples to training data.")

# 4. Retrain Student Model on Augmented Data
print("Phase 3: Retraining Student Model...")
X_augmented = pd.concat([X, pseudo_X])
y_augmented = pd.concat([y, pd.Series(pseudo_y)])

model_student = XGBClassifier(n_estimators=200, max_depth=4, random_state=42)
model_student.fit(X_augmented, y_augmented)

# 5. Final Prediction
final_probs = model_student.predict_proba(X_test)[:, 1]

# Thresholding
thresh = np.quantile(final_probs, 1 - (y.mean() * 2)) # Top N%
preds = (final_probs >= thresh).astype(int)

sub = pd.DataFrame({'object_id': test_meta['object_id'], 'prediction': preds})
sub.to_csv('submission_pseudolabel.csv', index=False)
print("Done.")
