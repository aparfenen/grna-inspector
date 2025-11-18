"""
Baseline machine learning models for gRNA classification.
Modified version with optional XGBoost support for Mac compatibility.

Implements:
1. Random Forest classifier (always available)
2. XGBoost classifier (optional, requires libomp on Mac)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import joblib
import json
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional XGBoost import
try:
    import xgboost as xgb
    from xgboost.callback import EarlyStopping
    XGBOOST_AVAILABLE = True
except (ImportError, OSError) as e:
    XGBOOST_AVAILABLE = False
    warnings.warn(f"⚠️  XGBoost not available: {e}\n"
                  "Only Random Forest will be used. "
                  "On Mac, run 'brew install libomp' to fix.")


class gRNABaselineClassifier:
    """Baseline ML classifier for gRNA identification."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            random_state: Random seed
        """
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            print("⚠️  XGBoost not available. Falling back to Random Forest.")
            print("   To use XGBoost on Mac: brew install libomp")
            model_type = 'random_forest'
        
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.results = {}
        
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names (exclude metadata)."""
        exclude_cols = ['sequence_id', 'label', 'source', 'sequence']
        return [col for col in df.columns if col not in exclude_cols]
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """
        Train the classifier.
        
        Args:
            train_df: Training dataframe with features and labels
            val_df: Optional validation dataframe for early stopping
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {self.model_type.upper()} CLASSIFIER")
        print(f"{'='*60}\n")
        
        # Prepare features and labels
        self.feature_names = self._get_feature_columns(train_df)
        X_train = train_df[self.feature_names].values
        y_train = train_df['label'].values
        
        print(f"Training set: {len(X_train)} samples, {len(self.feature_names)} features")
        print(f"Class distribution: {np.sum(y_train==1)} positive, {np.sum(y_train==0)} negative")
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost not available. Use 'random_forest' instead.")
            
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
            
            self.model = xgb.XGBClassifier(
                max_depth=10,
                learning_rate=0.1,
                n_estimators=300,  # Уменьшили с 500
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=1
            )
    
            # Simple fit - works with all XGBoost versions
            self.model.fit(X_train, y_train)
            print("  Note: Training without early stopping for compatibility")
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print("\nTraining complete!")
        
        # Evaluate on training set
        train_metrics = self.evaluate(train_df, dataset_name='Training')
        self.results['train'] = train_metrics
        
        # Evaluate on validation set if provided
        if val_df is not None:
            val_metrics = self.evaluate(val_df, dataset_name='Validation')
            self.results['val'] = val_metrics
    
    def evaluate(self, df: pd.DataFrame, dataset_name: str = 'Test') -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            df: Dataframe with features and labels
            dataset_name: Name of dataset for reporting
            
        Returns:
            Dictionary of metrics
        """
        X = df[self.feature_names].values
        y_true = df['label'].values
        
        # Predictions
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Print results
        print(f"\n{dataset_name} Results:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-score:    {metrics['f1']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type in ['random_forest', 'xgboost']:
            importances = self.model.feature_importances_
        else:
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Path = None):
        """Plot top N most important features."""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'].values)
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Most Important Features ({self.model_type})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, test_df: pd.DataFrame, save_path: Path = None):
        """Plot ROC curve."""
        X_test = test_df[self.feature_names].values
        y_test = test_df['label'].values
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_type}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, test_df: pd.DataFrame, save_path: Path = None):
        """Plot precision-recall curve."""
        X_test = test_df[self.feature_names].values
        y_test = test_df['label'].values
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_type}')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def save_model(self, save_path: Path):
        """Save trained model."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'results': self.results
        }
        
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: Path):
        """Load trained model."""
        model_data = joblib.load(load_path)
        
        classifier = cls(model_type=model_data['model_type'])
        classifier.model = model_data['model']
        classifier.feature_names = model_data['feature_names']
        classifier.results = model_data['results']
        
        print(f"Model loaded from {load_path}")
        return classifier
    
    def predict(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new sequences.
        
        Args:
            sequences: List of nucleotide sequences
            
        Returns:
            predictions (0/1), probabilities
        """
        from .data_preparation import gRNADataPreparator
        
        # Create temporary dataframe
        seq_dict = {f"seq_{i}": seq for i, seq in enumerate(sequences)}
        preparator = gRNADataPreparator(Path("../data"))
        features_df = preparator.compute_sequence_features(seq_dict)
        
        # Extract features in correct order
        X = features_df[self.feature_names].values
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities


def compare_models(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                  test_df: pd.DataFrame, save_dir: Path = None):
    """
    Train and compare available models (RF and optionally XGBoost).
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        save_dir: Directory to save models and plots
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")
    
    results = {}
    
    # Train Random Forest
    print("\n1. RANDOM FOREST")
    print("-" * 60)
    rf_classifier = gRNABaselineClassifier(model_type='random_forest')
    rf_classifier.train(train_df, val_df)
    rf_test_metrics = rf_classifier.evaluate(test_df, dataset_name='Test')
    results['random_forest'] = rf_test_metrics
    
    if save_dir:
        rf_classifier.save_model(save_dir / "rf_model.joblib")
        rf_classifier.plot_feature_importance(save_path=save_dir / "rf_feature_importance.png")
        rf_classifier.plot_roc_curve(test_df, save_path=save_dir / "rf_roc_curve.png")
    
    # Train XGBoost if available
    xgb_classifier = None
    if XGBOOST_AVAILABLE:
        print("\n2. XGBOOST")
        print("-" * 60)
        xgb_classifier = gRNABaselineClassifier(model_type='xgboost')
        xgb_classifier.train(train_df, val_df)
        xgb_test_metrics = xgb_classifier.evaluate(test_df, dataset_name='Test')
        results['xgboost'] = xgb_test_metrics
        
        if save_dir:
            xgb_classifier.save_model(save_dir / "xgb_model.joblib")
            xgb_classifier.plot_feature_importance(save_path=save_dir / "xgb_feature_importance.png")
            xgb_classifier.plot_roc_curve(test_df, save_path=save_dir / "xgb_roc_curve.png")
    else:
        print("\n2. XGBOOST")
        print("-" * 60)
        print("⚠️  XGBoost not available. Skipping.")
        print("   To enable XGBoost on Mac: brew install libomp")
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60 + "\n")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))
    
    if save_dir:
        comparison_df.to_csv(save_dir / "model_comparison.csv")
        
        # Save results as JSON
        with open(save_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    return rf_classifier, xgb_classifier, results


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    data_dir = Path("../data/processed")
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(data_dir / "train_data.csv")
    val_df = pd.read_csv(data_dir / "val_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")
    
    # Compare models
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    rf_model, xgb_model, results = compare_models(
        train_df, val_df, test_df, 
        save_dir=models_dir
    )
    
    print("\nModel comparison complete!")
