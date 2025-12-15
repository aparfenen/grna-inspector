"""
gRNA Classification: Model Training v4.0

Works with data from 2_data_preparation_v4.py
Uses v4_pipeline directory with corrected features and clean splits.
"""

import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
np.random.seed(42)

print("✓ Imports loaded")
print(f"  XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not available'}")
print(f"  SHAP: {'Available' if SHAP_AVAILABLE else 'Not available'}")


# =============================================================================
# FILE PATHS - UPDATE FOR YOUR ENVIRONMENT
# =============================================================================

PROJECT_ROOT = Path.home() / 'projects' / 'grna-inspector'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'v4_pipeline'  # v4!
MODELS_DIR = PROJECT_ROOT / 'models'
PLOTS_DIR = PROJECT_ROOT / 'data' / 'plots' / 'model_analysis_v4'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nData directory: {DATA_DIR}")
print(f"Models directory: {MODELS_DIR}")

# Check files
required_files = ['train_data.csv', 'val_data.csv', 'test_data.csv', 'feature_names.txt']
for f in required_files:
    path = DATA_DIR / f
    status = "[OK]" if path.exists() else "[MISSING]"
    print(f"  {status} {f}")


# =============================================================================
# STAGE 1: LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 1: LOAD & VERIFY DATA")
print("=" * 80)

train_df = pd.read_csv(DATA_DIR / 'train_data.csv')
val_df = pd.read_csv(DATA_DIR / 'val_data.csv')
test_df = pd.read_csv(DATA_DIR / 'test_data.csv')

print(f"\nDataset sizes:")
print(f"  Train: {len(train_df):,}")
print(f"  Val:   {len(val_df):,}")
print(f"  Test:  {len(test_df):,}")

with open(DATA_DIR / 'feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"\nFeatures: {len(feature_names)}")

# Verify no length leakage
if 'length' in feature_names:
    print("[WARNING] 'length' in features - removing!")
    feature_names.remove('length')
else:
    print("[OK] No raw length feature")

# Class distribution
print("\nClass distribution:")
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    pos = sum(df['label'] == 1)
    neg = sum(df['label'] == 0)
    ratio = min(pos, neg) / max(pos, neg)
    print(f"  {name}: pos={pos:,} neg={neg:,} ratio={ratio:.3f}")


# =============================================================================
# PREPARE FEATURE MATRICES
# =============================================================================

X_train = train_df[feature_names].values
y_train = train_df['label'].values

X_val = val_df[feature_names].values
y_val = val_df['label'].values

X_test = test_df[feature_names].values
y_test = test_df['label'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"\nFeature matrices: X_train={X_train.shape}")


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, X, y, name="Model"):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_prob) if y_prob is not None else 0
    }
    return metrics


def print_metrics(metrics, name="Model"):
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")


# =============================================================================
# STAGE 2: BASELINE MODELS
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 2: BASELINE MODELS")
print("=" * 80)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)

baseline_results = {}
best_model_name = None
best_val_auc = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        train_metrics = evaluate_model(model, X_train_scaled, y_train)
        val_metrics = evaluate_model(model, X_val_scaled, y_val)
    else:
        model.fit(X_train, y_train)
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
    
    baseline_results[name] = {'train': train_metrics, 'val': val_metrics}
    print_metrics(val_metrics, f"{name} (Val)")
    
    if val_metrics['roc_auc'] > best_val_auc:
        best_val_auc = val_metrics['roc_auc']
        best_model_name = name

print(f"\n🏆 Best baseline: {best_model_name} (ROC-AUC: {best_val_auc:.4f})")


# =============================================================================
# STAGE 3: HYPERPARAMETER TUNING
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 3: HYPERPARAMETER TUNING")
print("=" * 80)

# Tune Random Forest
print("\nTuning Random Forest...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print(f"  Best params: {rf_grid.best_params_}")
print(f"  Best CV score: {rf_grid.best_score_:.4f}")

best_rf = rf_grid.best_estimator_
rf_val_metrics = evaluate_model(best_rf, X_val, y_val)
print_metrics(rf_val_metrics, "Tuned RF (Val)")

# Tune XGBoost if available
if XGBOOST_AVAILABLE:
    print("\nTuning XGBoost...")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42, n_jobs=-1),
        xgb_param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    
    print(f"  Best params: {xgb_grid.best_params_}")
    print(f"  Best CV score: {xgb_grid.best_score_:.4f}")
    
    best_xgb = xgb_grid.best_estimator_
    xgb_val_metrics = evaluate_model(best_xgb, X_val, y_val)
    print_metrics(xgb_val_metrics, "Tuned XGBoost (Val)")

# Select final model
if XGBOOST_AVAILABLE and xgb_val_metrics['roc_auc'] > rf_val_metrics['roc_auc']:
    final_model = best_xgb
    final_model_name = 'XGBoost'
else:
    final_model = best_rf
    final_model_name = 'Random Forest'

print(f"\n🏆 Final model: {final_model_name}")


# =============================================================================
# STAGE 4: TEST SET EVALUATION
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 4: TEST SET EVALUATION")
print("=" * 80)

test_metrics = evaluate_model(final_model, X_test, y_test)
print_metrics(test_metrics, f"{final_model_name} (TEST)")

# Confusion matrix
y_pred = final_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

# Classification report
print(f"\n{classification_report(y_test, y_pred, target_names=['non-gRNA', 'gRNA'])}")


# =============================================================================
# STAGE 5: FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 5: FEATURE IMPORTANCE")
print("=" * 80)

# MDI importance
if hasattr(final_model, 'feature_importances_'):
    mdi_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 20 Features (MDI):")
    print("-" * 50)
    for i, row in mdi_importance.head(20).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:.4f}")

# Permutation importance
print("\nCalculating permutation importance...")
perm_imp = permutation_importance(final_model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': perm_imp.importances_mean
}).sort_values('importance', ascending=False)

print("\n📊 Top 20 Features (Permutation):")
print("-" * 50)
for i, row in perm_importance.head(20).iterrows():
    print(f"  {row['feature']:<35} {row['importance']:.4f}")


# =============================================================================
# STAGE 6: BIOLOGICAL VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 6: BIOLOGICAL VALIDATION")
print("=" * 80)

bio_features = [
    ('anchor_AC_content', 'HIGH', 'mRNA binding'),
    ('anchor_AC_rich', 'HIGH', 'AC > 60%'),
    ('anchor_G_depleted', 'HIGH', 'G < 15%'),
    ('init_any_known_pattern', 'HIGH', 'ATATA/AWAHH'),
    ('init_starts_A', 'HIGH', 'A start'),
    ('guide_A_elevated', 'MEDIUM', 'A > 40%'),
    ('no_palindrome_5bp', 'HIGH', 'Open structure'),
    ('in_molecular_ruler_range', 'MEDIUM', '15-19 nt'),
    ('ends_with_T', 'MEDIUM', 'U-tail site'),
]

print("\n🧬 Biological Feature Ranking:")
print("-" * 60)
print(f"{'Feature':<30} {'Expected':>10} {'MDI Rank':>10} {'Perm Rank':>10}")
print("-" * 60)

for feat, expected, desc in bio_features:
    if feat in mdi_importance['feature'].values:
        mdi_rank = mdi_importance[mdi_importance['feature'] == feat].index[0] + 1
        perm_rank = perm_importance[perm_importance['feature'] == feat].index[0] + 1
        print(f"{feat:<30} {expected:>10} {mdi_rank:>10} {perm_rank:>10}")


# =============================================================================
# STAGE 7: SAVE MODEL
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 7: SAVE MODEL")
print("=" * 80)

model_file = MODELS_DIR / 'grna_classifier_v4.joblib'
joblib.dump(final_model, model_file)
print(f"  ✓ {model_file}")

scaler_file = MODELS_DIR / 'scaler_v4.joblib'
joblib.dump(scaler, scaler_file)
print(f"  ✓ {scaler_file}")

# Save results
results = {
    'model': final_model_name,
    'pipeline_version': '4.0',
    'test_metrics': test_metrics,
    'feature_count': len(feature_names),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'top_features_mdi': mdi_importance.head(20).to_dict('records'),
    'top_features_perm': perm_importance.head(20).to_dict('records'),
}

results_file = MODELS_DIR / 'training_results_v4.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  ✓ {results_file}")


print("\n" + "=" * 80)
print("✅ MODEL TRAINING v4.0 COMPLETE!")
print("=" * 80)
print(f"\n📊 Test Results:")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1-score:  {test_metrics['f1']:.4f}")
print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
print(f"\n📁 Model saved to: {model_file}")
