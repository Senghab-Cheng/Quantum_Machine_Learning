import pandas as pd
import numpy as np
import os

def load_data():
    """
    Load the diabetes dataset from the data folder
    Returns:
        X: Features (all columns except 'Outcome')
        y: Target ('Outcome' column)
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root, then into data folder
    csv_path = os.path.join(script_dir, '..', 'data', 'diabetes.csv')
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find diabetes.csv at {csv_path}")
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

def load_data_with_preprocessing():
    """
    Load and preprocess the diabetes dataset
    Returns:
        X_scaled: Preprocessed features
        y: Target
        scaler: Fitted scaler for later use
    """
    X, y = load_data()
    
    # Simple preprocessing: scale the features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

if __name__ == "__main__":
    # Test the data loader
    X, y = load_data()
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nFirst 5 rows of features:")
    print(X.head())
    print("\nFirst 5 target values:")
    print(y.head())