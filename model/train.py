"""
Model Training Script for Employee Attrition Prediction
This script trains a RandomForestClassifier to predict employee attrition.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import json
import os
from datetime import datetime

# Configuration
DATA_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "attrition_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")


def load_data(url: str) -> pd.DataFrame:
    """Load the employee attrition dataset."""
    print(f"📊 Loading data from {url}...")
    df = pd.read_csv(url)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data:
    - Drop irrelevant columns
    - Encode categorical variables
    - Separate features and target
    """
    print("\n🔧 Preprocessing data...")
    
    # Drop columns that don't add predictive value
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop(columns=cols_to_drop)
    print(f"   Dropped columns: {cols_to_drop}")
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include='object').columns
    
    encodings = {}
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        encodings[col] = {label: idx for idx, label in enumerate(le.classes_)}
    
    print(f"   Encoded {len(categorical_cols)} categorical columns")
    
    # Separate features and target
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, X.columns.tolist(), encodings


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train a RandomForestClassifier with cross-validation.
    """
    print("\n🤖 Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    # Perform cross-validation
    print("\n📈 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"   CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    print("✅ Model training complete")
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n🎯 Test Set Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))
    
    return model, {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'cv_roc_auc_mean': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std(),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }


def save_artifacts(model, feature_columns: list, metrics: dict, encodings: dict):
    """Save model, feature columns, and metrics to disk."""
    print("\n💾 Saving artifacts...")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"   Model saved to: {MODEL_PATH}")
    
    # Save feature columns
    with open(FEATURE_COLUMNS_PATH, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    print(f"   Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    
    # Save encodings
    encodings_path = os.path.join(MODEL_DIR, "encodings.json")
    with open(encodings_path, 'w') as f:
        json.dump(encodings, f, indent=2)
    print(f"   Encodings saved to: {encodings_path}")
    
    # Save metrics
    metrics['timestamp'] = datetime.now().isoformat()
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to: {METRICS_PATH}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("🚀 Employee Attrition Model Training Pipeline")
    print("=" * 60)
    
    # Load data
    df = load_data(DATA_URL)
    
    # Preprocess
    X, y, feature_columns, encodings = preprocess_data(df)
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Save artifacts
    save_artifacts(model, feature_columns, metrics, encodings)
    
    print("\n" + "=" * 60)
    print("✅ Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
