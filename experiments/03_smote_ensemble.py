import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

# Load Data (Assume pre-processed)
train_final = pd.read_csv('processed_train_data.csv')
X = train_final.drop(columns=['object_id', 'target'])
y = train_final['target']

# Handle NaN values for Trees that don't support them natively
imputer = SimpleImputer(strategy='constant', fill_value=-999)
X_imp = imputer.fit_transform(X)

# --- Strategy 1: SMOTE (Synthetic Data) ---
# Upsample the minority class (TDEs) to 20% of the majority
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_res, y_res = smote.fit_resample(X_imp, y)

# --- Strategy 2: Voting Ensemble ---
clf1 = HistGradientBoostingClassifier(random_state=42)
clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
clf3 = ExtraTreesClassifier(n_estimators=200, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('hgb', clf1), ('rf', clf2), ('et', clf3)],
    voting='soft'
)

print("Training Ensemble on SMOTE-augmented data...")
voting_clf.fit(X_res, y_res)
print("Training Complete.")
