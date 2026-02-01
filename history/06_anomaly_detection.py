import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================================
# EXPERIMENT: UNSUPERVISED ANOMALY DETECTION
# Hypothesis: TDEs are rare events (<1%). We can find them by identifying
#             statistical outliers without using labels.
# ==========================================

# 1. Load Data
train_lc = pd.read_csv('../data/train_lightcurves.csv')
train_meta = pd.read_csv('../data/train_log.csv')
test_lc = pd.read_csv('../data/test_lightcurves.csv')
test_meta = pd.read_csv('../data/test_log.csv')

def get_features(df):
    # Basic Stats + Peak Flux
    return df.groupby('object_id')['Flux'].agg(['mean', 'std', 'max', 'min']).reset_index()

# We combine Train and Test because Unsupervised models love more data
print("Preparing Data...")
train_feat = get_features(train_lc)
test_feat = get_features(test_lc)
full_data = pd.concat([train_feat, test_feat], ignore_index=True)
ids = full_data['object_id']

# 2. Preprocessing
# Isolation Forest requires scaled data to calculate distances correctly
scaler = StandardScaler()
X = scaler.fit_transform(full_data.drop(columns=['object_id']))

# 3. Train Isolation Forest
print("Training Isolation Forest...")
# contamination=0.01 means "We expect 1% of the data to be anomalies"
iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X)

# 4. Predict
# -1 = Anomaly, 1 = Normal
preds = iso.predict(X)

# 5. Extract Test Predictions
full_data['anomaly_score'] = preds
full_data['object_id'] = ids

# Filter for Test Set only
test_results = full_data[full_data['object_id'].isin(test_meta['object_id'])]

# Convert -1 (Anomaly) to 1 (Target), and 1 (Normal) to 0
final_preds = np.where(test_results['anomaly_score'] == -1, 1, 0)

sub = pd.DataFrame({'object_id': test_results['object_id'], 'prediction': final_preds})
sub.to_csv('submission_anomaly_physics.csv', index=False)
print(f"Anomaly Detection Complete. Found {final_preds.sum()} potential TDEs.")
