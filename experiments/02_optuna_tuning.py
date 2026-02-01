import pandas as pd
import optuna
import sys
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.physics_features import extract_features

# Load and Process Data (Assume pre-processed X and y are available)
# For this script, we assume 'train_final.csv' is saved for speed
train_final = pd.read_csv('processed_train_data.csv') 
X = train_final.drop(columns=['object_id', 'target'])
y = train_final['target']

def objective(trial):
    """
    Optuna Objective Function to optimize XGBoost Hyperparameters
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': 19.5, # Calculated from Class Imbalance
        'tree_method': 'hist',    # Use 'gpu_hist' if GPU is available
        'random_state': 42
    }
    
    # 3-Fold CV for speed
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds))
        
    return sum(f1_scores) / len(f1_scores)

# Run Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"Best Trial: {study.best_params}")
