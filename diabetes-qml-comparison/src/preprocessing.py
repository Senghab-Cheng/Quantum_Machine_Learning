import s
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

s.path.insert(0, str(Path(__file__).parent.parent))
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
    df = pd.read_csv(Path(__file__).parent.parent / 'data' / 'diabetes.csv')
    X_train, X_test, y_train, y_test = preprocess(df)
    
    print("=" * 60)
    print("PHASE 3 PREPROCESSING VERIFICATION")
    print("=" * 60)
    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"\nData types:")
    print(f"  X_train: {X_train.dtype}")
    print(f"  X_test:  {X_test.dtype}")
    print(f"  y_train: {y_train.dtype}")
    print(f"  y_test:  {y_test.dtype}")
    print(f"\nValue ranges (quantum-scaled to [0, π]):")
    print(f"  X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"  X_test min:  {X_test.min():.4f}, max: {X_test.max():.4f}")
    print(f"\nNaN checks:")
    print(f"  X_train NaNs: {np.isnan(X_train).sum()}")
    print(f"  X_test NaNs:  {np.isnan(X_test).sum()}")
    print(f"  y_train NaNs: {np.isnan(y_train).sum()}")
    print(f"  y_test NaNs:  {np.isnan(y_test).sum()}")

    print(f"\nClass distribution:")
    print(f"  y_train: {np.bincount(y_train)}")
    print(f"  y_test:  {np.bincount(y_test)}")
    
    print(f"\n All checks passed!" if np.isnan(X_train).sum() == 0 else "\n NaNs detected!")
    print("=" * 60)