"""
Generate sample test data for the Streamlit app
This creates a sample CSV file that can be uploaded to test the application
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_sample_test_data(n_samples=200, n_features=11, output_file='sample_test_data.csv'):
    """
    Generate synthetic classification data for testing
    
    Parameters:
    - n_samples: Number of samples (minimum 500 for assignment)
    - n_features: Number of features (minimum 12 for assignment)
    - output_file: Name of output CSV file
    """
    
    print("Generating sample test data...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.1,
        random_state=42,
        class_sep=0.8
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Sample data saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Features: {n_features}")
    print(f"Samples: {n_samples}")
    print(f"\nClass Distribution:")
    print(df['target'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

def generate_wine_quality_sample(n_samples=200):
    """
    Generate wine quality-like synthetic data
    """
    
    print("Generating wine quality sample data...")
    
    np.random.seed(42)
    
    data = {
        'fixed_acidity': np.random.uniform(4, 16, n_samples),
        'volatile_acidity': np.random.uniform(0.1, 1.6, n_samples),
        'citric_acid': np.random.uniform(0, 1, n_samples),
        'residual_sugar': np.random.uniform(0.9, 15, n_samples),
        'chlorides': np.random.uniform(0.01, 0.6, n_samples),
        'free_sulfur_dioxide': np.random.uniform(1, 70, n_samples),
        'total_sulfur_dioxide': np.random.uniform(6, 290, n_samples),
        'density': np.random.uniform(0.99, 1.01, n_samples),
        'pH': np.random.uniform(2.7, 4.0, n_samples),
        'sulphates': np.random.uniform(0.3, 2.0, n_samples),
        'alcohol': np.random.uniform(8, 15, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target based on some logic
    # Good wine (1) if: high alcohol, low volatile acidity, moderate pH
    df['quality'] = (
        (df['alcohol'] > 10.5) & 
        (df['volatile_acidity'] < 0.6) & 
        (df['pH'] > 3.0) & 
        (df['pH'] < 3.6)
    ).astype(int)
    
    # Add some randomness
    random_flips = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    df['quality'] = np.where(random_flips == 1, 1 - df['quality'], df['quality'])
    
    # Save to CSV
    output_file = 'wine_quality_sample.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Wine quality sample saved to: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"\nClass Distribution:")
    print(df['quality'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    print("="*60)
    print("SAMPLE DATA GENERATOR")
    print("="*60)
    
    # Generate generic classification data
    print("\n1. Generating generic classification data...")
    df1 = generate_sample_test_data(n_samples=200, n_features=11)
    
    print("\n" + "="*60)
    
    # Generate wine quality-like data
    print("\n2. Generating wine quality sample data...")
    df2 = generate_wine_quality_sample(n_samples=200)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print("\nYou can now upload these files to test the Streamlit app:")
    print("1. sample_test_data.csv")
    print("2. wine_quality_sample.csv")
