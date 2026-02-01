"""
Generate multiple test datasets for ML Assignment 2
Creates various CSV files you can upload to test the Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def create_wine_quality_real_format():
    """
    Create a dataset in real Wine Quality format with proper features
    This mimics the actual UCI Wine Quality dataset
    """
    np.random.seed(42)
    n_samples = 1599
    
    # Generate realistic wine quality features
    data = {
        'fixed acidity': np.random.uniform(4.6, 15.9, n_samples),
        'volatile acidity': np.random.uniform(0.12, 1.58, n_samples),
        'citric acid': np.random.uniform(0, 1, n_samples),
        'residual sugar': np.random.uniform(0.9, 15.5, n_samples),
        'chlorides': np.random.uniform(0.012, 0.611, n_samples),
        'free sulfur dioxide': np.random.uniform(1, 72, n_samples),
        'total sulfur dioxide': np.random.uniform(6, 289, n_samples),
        'density': np.random.uniform(0.99007, 1.00369, n_samples),
        'pH': np.random.uniform(2.74, 4.01, n_samples),
        'sulphates': np.random.uniform(0.33, 2.0, n_samples),
        'alcohol': np.random.uniform(8.4, 14.9, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create quality score based on features (binary classification)
    # Good wine (1) if: high alcohol, low volatile acidity, moderate pH
    quality_score = (
        (df['alcohol'] > 10.5) * 2 +
        (df['volatile acidity'] < 0.6) * 2 +
        (df['pH'] > 3.0) * 1 +
        (df['pH'] < 3.6) * 1 +
        (df['citric acid'] > 0.3) * 1 +
        (df['sulphates'] > 0.6) * 1
    )
    
    # Convert to binary (0 or 1)
    df['quality'] = (quality_score >= 4).astype(int)
    
    # Add some randomness (15% noise)
    random_flips = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    df['quality'] = np.where(random_flips == 1, 1 - df['quality'], df['quality'])
    
    return df

def create_small_test_dataset():
    """
    Create a small dataset for quick testing (200 samples)
    """
    X, y = make_classification(
        n_samples=200,
        n_features=11,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=42,
        class_sep=0.8
    )
    
    feature_names = [f'feature_{i+1}' for i in range(11)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def create_large_test_dataset():
    """
    Create a larger dataset for comprehensive testing (1000 samples)
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=11,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=3,
        flip_y=0.08,
        random_state=123,
        class_sep=1.0
    )
    
    feature_names = [f'feature_{i+1}' for i in range(11)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def create_balanced_dataset():
    """
    Create perfectly balanced dataset (50-50 class distribution)
    """
    X, y = make_classification(
        n_samples=600,
        n_features=11,
        n_informative=9,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        flip_y=0.05,
        random_state=77,
        class_sep=1.2
    )
    
    feature_names = [f'feature_{i+1}' for i in range(11)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def create_imbalanced_dataset():
    """
    Create imbalanced dataset (80-20 class distribution)
    """
    X, y = make_classification(
        n_samples=500,
        n_features=11,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.8, 0.2],
        flip_y=0.1,
        random_state=99,
        class_sep=0.7
    )
    
    feature_names = [f'feature_{i+1}' for i in range(11)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def main():
    """Generate all test datasets"""
    
    print("="*70)
    print("GENERATING TEST DATASETS FOR ML ASSIGNMENT 2")
    print("="*70)
    
    datasets = {
        '1_wine_quality_full.csv': (create_wine_quality_real_format, "Full Wine Quality Dataset (1599 samples)"),
        '2_small_test.csv': (create_small_test_dataset, "Small Test Dataset (200 samples)"),
        '3_large_test.csv': (create_large_test_dataset, "Large Test Dataset (1000 samples)"),
        '4_balanced.csv': (create_balanced_dataset, "Balanced Dataset (50-50 split)"),
        '5_imbalanced.csv': (create_imbalanced_dataset, "Imbalanced Dataset (80-20 split)"),
    }
    
    for filename, (func, description) in datasets.items():
        print(f"\n📊 Creating: {filename}")
        print(f"   {description}")
        
        df = func()
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Saved: {filename}")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {df.shape[1] - 1}")
        print(f"   Samples: {df.shape[0]}")
        print(f"   Class Distribution:")
        target_col = 'quality' if 'quality' in df.columns else 'target'
        print(f"      {df[target_col].value_counts().to_dict()}")
        print(f"   First few rows:")
        print(df.head(3).to_string())
    
    print("\n" + "="*70)
    print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Files:")
    for filename in datasets.keys():
        print(f"  ✓ {filename}")
    
    print("\n💡 Upload any of these files to your Streamlit app!")
    print("   Recommended: Start with '1_wine_quality_full.csv'")

if __name__ == "__main__":
    main()
