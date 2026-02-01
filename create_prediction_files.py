"""
Create prediction-ready test files
This script removes the target column from test datasets to create prediction files
"""

import pandas as pd
import os

print("="*70)
print("CREATING PREDICTION-READY FILES")
print("="*70)

# Files to process
files_to_convert = [
    '1_wine_quality_full.csv',
    '2_small_test.csv',
    '3_large_test.csv',
    '4_balanced.csv',
    '5_imbalanced.csv'
]

output_dir = 'prediction_files'
os.makedirs(output_dir, exist_ok=True)

for filename in files_to_convert:
    if os.path.exists(filename):
        print(f"\n📄 Processing: {filename}")
        
        # Read file
        df = pd.read_csv(filename)
        
        # Remove target column (last column)
        df_pred = df.iloc[:, :-1]
        
        # Create output filename
        output_name = filename.replace('.csv', '_PREDICTION.csv')
        output_path = os.path.join(output_dir, output_name)
        
        # Save
        df_pred.to_csv(output_path, index=False)
        
        print(f"   ✅ Created: {output_path}")
        print(f"   Shape: {df_pred.shape[0]} samples, {df_pred.shape[1]} features")
        print(f"   Features: {', '.join(df_pred.columns[:3])}...")
    else:
        print(f"\n   ⚠️  File not found: {filename}")

print("\n" + "="*70)
print("DONE! Prediction files created in 'prediction_files/' folder")
print("="*70)
print("\n💡 How to use:")
print("1. Train models using original files (with target column)")
print("2. Use files in 'prediction_files/' folder for making predictions")
print("3. These files have the same features but without target column")
