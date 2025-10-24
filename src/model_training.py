"""
Model Training Script
---------------------
Train and evaluate ML models for accident severity prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(data_path='data/processed/'):
    """
    Load preprocessed data
    
    Parameters:
    -----------
    data_path : str
        Path to processed data
    
    Returns:
    --------
    tuple
        X, y dataframes
    """
    print("📂 Loading processed data...")
    
    X = pd.read_csv(f"{data_path}/X_features.csv")
    y = pd.read_csv(f"{data_path}/y_target.csv").values.ravel()
    
    print(f"✅ Loaded data: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"   Target classes: {np.unique(y)}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : np.array
        Target
    test_size : float
        Proportion of test data
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    print(f"\n📊 Splitting data (train: {(1-test_size)*100}%, test: {test_size*100}%)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scale numerical features
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    
    Returns:
    --------
    tuple
        X_train_scaled, X_test_scaled, scaler
    """
    print("\n🔧 Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Features scaled")
    
    return X_train_scaled, X_test_scaled, scaler


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest classifier
    
    Returns:
    --------
    tuple
        model, predictions, metrics
    """
    print("\n" + "="*60)
    print("🌲 TRAINING RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    # Initialize model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    # Train
    print("⏳ Training...")
    start_time = datetime.now()
    rf_model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    metrics['train_time'] = train_time
    
    print(f"\n✅ Random Forest trained in {train_time:.2f} seconds")
    
    return rf_model, y_pred, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier
    
    Returns:
    --------
    tuple
        model, predictions, metrics
    """
    if not XGBOOST_AVAILABLE:
        print("\n⚠️  XGBoost not available, skipping...")
        return None, None, None
    
    print("\n" + "="*60)
    print("⚡ TRAINING XGBOOST CLASSIFIER")
    print("="*60)
    
    # Convert labels to 0-indexed for XGBoost
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Initialize model
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # Train
    print("⏳ Training...")
    start_time = datetime.now()
    xgb_model.fit(X_train, y_train_encoded)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred_encoded = xgb_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)
    y_pred_proba = xgb_model.predict_proba(X_test)
    
    # Evaluate (use original labels)
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
    metrics['train_time'] = train_time
    
    print(f"\n✅ XGBoost trained in {train_time:.2f} seconds")
    
    return xgb_model, y_pred, metrics


def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train Decision Tree classifier (baseline)
    
    Returns:
    --------
    tuple
        model, predictions, metrics
    """
    print("\n" + "="*60)
    print("🌳 TRAINING DECISION TREE CLASSIFIER (Baseline)")
    print("="*60)
    
    # Initialize model
    dt_model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=20,
        random_state=42
    )
    
    # Train
    print("⏳ Training...")
    start_time = datetime.now()
    dt_model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred = dt_model.predict(X_test)
    y_pred_proba = dt_model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Decision Tree")
    metrics['train_time'] = train_time
    
    print(f"\n✅ Decision Tree trained in {train_time:.2f} seconds")
    
    return dt_model, y_pred, metrics


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    y_pred_proba : array
        Prediction probabilities
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict
        Metrics dictionary
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Print results
    print(f"\n📊 {model_name} Performance:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"   {cm}")
    
    # Store metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics


def compare_models(results):
    """
    Compare all trained models
    
    Parameters:
    -----------
    results : list
        List of (model_name, metrics) tuples
    """
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for model_name, metrics in results:
        if metrics is not None:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']*100:.2f}%",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Train Time': f"{metrics['train_time']:.2f}s"
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))
    
    # Find best model
    best_model = max(results, key=lambda x: x[1]['accuracy'] if x[1] else 0)
    print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']*100:.2f}%)")
    
    return df_comparison, best_model[0]


def plot_confusion_matrix(cm, model_name, save_path='assets/'):
    """
    Plot and save confusion matrix
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Minor', 'Moderate', 'Severe', 'Fatal'],
                yticklabels=['Minor', 'Moderate', 'Severe', 'Fatal'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"{save_path}/{model_name.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved confusion matrix: {filename}")


def save_model(model, scaler, model_name, save_path='models/'):
    """
    Save trained model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    scaler : StandardScaler
        Feature scaler
    model_name : str
        Name for saving
    save_path : str
        Directory to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_filename = f"{save_path}/{model_name.replace(' ', '_').lower()}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_filename = f"{save_path}/scaler.pkl"
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"💾 Saved model: {model_filename}")
    print(f"💾 Saved scaler: {scaler_filename}")


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("🚦 TRAFFIC ACCIDENT SEVERITY PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Check if processed data exists
    if not os.path.exists('data/processed/X_features.csv'):
        print("\n⚠️  Processed data not found!")
        print("   Please run: python src/data_preprocessing.py")
        return
    
    # Load data
    X, y = load_processed_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features (optional, but helps some models)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train models
    results = []
    
    # 1. Random Forest (works well without scaling)
    rf_model, rf_pred, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(("Random Forest", rf_metrics))
    plot_confusion_matrix(rf_metrics['confusion_matrix'], "Random Forest")
    
    # 2. XGBoost (if available)
    xgb_model, xgb_pred, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_metrics:
        results.append(("XGBoost", xgb_metrics))
        plot_confusion_matrix(xgb_metrics['confusion_matrix'], "XGBoost")
    
    # 3. Decision Tree (baseline)
    dt_model, dt_pred, dt_metrics = train_decision_tree(X_train, y_train, X_test, y_test)
    results.append(("Decision Tree", dt_metrics))
    
    # Compare models
    comparison_df, best_model_name = compare_models(results)
    
    # Save comparison
    os.makedirs('assets', exist_ok=True)
    comparison_df.to_csv('assets/model_comparison.csv', index=False)
    print(f"\n💾 Model comparison saved to: assets/model_comparison.csv")
    
    # Save best model
    if best_model_name == "Random Forest":
        save_model(rf_model, scaler, "random_forest")
    elif best_model_name == "XGBoost":
        save_model(xgb_model, scaler, "xgboost")
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\n📌 Next steps:")
    print("   1. Run dashboard: streamlit run dashboard/app.py")
    print("   2. Or explore notebooks for detailed analysis")


if __name__ == "__main__":
    main()
