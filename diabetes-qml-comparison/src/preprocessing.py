import sys  # <--- Added this import
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Changed s.path to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_QUBITS, TEST_SIZE, RANDOM_STATE

def preprocess(df):
    X = df.drop('Outcome', axis=1).copy()
    y = df['Outcome'].copy()
    
    columns_with_impossible_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_with_impossible_zeros:
        if col in X.columns:
            median_val = X[col][X[col] > 0].median()
            X.loc[X[col] == 0, col] = median_val
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = (X_scaled - X_scaled.min(axis=0)) / (X_scaled.max(axis=0) - X_scaled.min(axis=0) + 1e-8)
    
    pca = PCA(n_components=N_QUBITS)
    X_pca = pca.fit_transform(X_scaled)
    
    # Rescaling to [0, 1] then mapping to [0, π]
    X_pca = (X_pca - X_pca.min(axis=0)) / (X_pca.max(axis=0) - X_pca.min(axis=0) + 1e-8)
    X_quantum = X_pca * np.pi
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_quantum, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Ensure the path to the CSV is correct relative to this file
    data_path = Path(__file__).parent.parent / 'data' / 'diabetes.csv'
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = preprocess(df)
    
    print("=" * 60)
    print("PHASE 3 PREPROCESSING VERIFICATION")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"NaN check: {np.isnan(X_train).sum()} NaNs found")
    print("=" * 60)