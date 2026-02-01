"""
Machine Learning Classification Models Implementation
BITS M.Tech Assignment 2
Wine Quality Dataset Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the Wine Quality dataset
def load_wine_quality_data():
    """
    Load Wine Quality Dataset
    Source: UCI Machine Learning Repository
    
    Features: 11 physicochemical properties
    Target: Wine quality (binary: good=1, bad=0)
    """
    # For demonstration, creating a sample dataset structure
    # In actual implementation, download from: 
    # https://archive.ics.uci.edu/ml/datasets/wine+quality
    
    print("Loading Wine Quality Dataset...")
    
    # Load red wine dataset (you should download this)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Samples: {df.shape[0]}")
    
    # Convert to binary classification
    # Quality >= 6 is good wine (1), otherwise bad wine (0)
    df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)
    
    print(f"\nClass Distribution:")
    print(df['quality'].value_counts())
    
    return df

# Train and evaluate all models
def train_and_evaluate_models(df):
    """
    Train 6 classification models and evaluate them
    """
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining Set: {X_train_scaled.shape}")
    print(f"Test Set: {X_test_scaled.shape}")
    
    # Define all 6 models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING MODELS")
    print("="*80)
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Get probability predictions for AUC
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = y_pred  # Fallback for models without predict_proba
        
        # Calculate all metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_pred_proba) if hasattr(model, 'predict_proba') else None,
            'Precision': precision_score(y_test, y_pred, average='binary'),
            'Recall': recall_score(y_test, y_pred, average='binary'),
            'F1': f1_score(y_test, y_pred, average='binary'),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        
        # Store results
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print metrics
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"{metric_name:12s}: {value:.4f}")
            else:
                print(f"{metric_name:12s}: N/A")
    
    # Save models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80)
    
    for model_name, result in results.items():
        filename = f"model_{model_name.replace(' ', '_').lower()}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"Saved: {filename}")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved: scaler.pkl")
    
    return results, scaler

# Display comparison table
def display_comparison_table(results):
    """
    Display comparison table of all models
    """
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'ML Model Name': model_name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'AUC': f"{metrics['AUC']:.4f}" if metrics['AUC'] is not None else "N/A",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1': f"{metrics['F1']:.4f}",
            'MCC': f"{metrics['MCC']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nComparison table saved to: model_comparison.csv")
    
    return comparison_df

# Generate observations
def generate_observations(results):
    """
    Generate observations about model performance
    """
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE OBSERVATIONS")
    print("="*80)
    
    observations = {}
    
    for model_name, result in results.items():
        metrics = result['metrics']
        
        # Generate observation based on metrics
        acc = metrics['Accuracy']
        f1 = metrics['F1']
        mcc = metrics['MCC']
        
        if acc >= 0.85 and f1 >= 0.85:
            performance = "Excellent"
            obs = f"Shows {performance} performance with high accuracy ({acc:.4f}) and balanced precision-recall (F1: {f1:.4f}). Strong choice for this dataset."
        elif acc >= 0.75 and f1 >= 0.75:
            performance = "Good"
            obs = f"Demonstrates {performance} performance with accuracy of {acc:.4f} and F1 score of {f1:.4f}. Reliable for predictions."
        elif acc >= 0.65:
            performance = "Moderate"
            obs = f"Shows {performance} performance with accuracy of {acc:.4f}. May need hyperparameter tuning for better results."
        else:
            performance = "Below Average"
            obs = f"Performance is {performance} with accuracy of {acc:.4f}. Consider feature engineering or different algorithm."
        
        observations[model_name] = obs
        print(f"\n{model_name}:")
        print(f"  {obs}")
    
    return observations

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("MACHINE LEARNING CLASSIFICATION - ASSIGNMENT 2")
    print("BITS Pilani M.Tech (AIML/DSE)")
    print("="*80)
    
    # Load dataset
    df = load_wine_quality_data()
    
    # Train and evaluate models
    results, scaler = train_and_evaluate_models(df)
    
    # Display comparison table
    comparison_df = display_comparison_table(results)
    
    # Generate observations
    observations = generate_observations(results)
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nNext Steps:")
    print("1. Review model_comparison.csv for detailed metrics")
    print("2. Check saved model files (*.pkl)")
    print("3. Run Streamlit app: streamlit run app.py")
    print("4. Deploy to Streamlit Cloud")
