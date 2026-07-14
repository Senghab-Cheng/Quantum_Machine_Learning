import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_data
from preprocessing import preprocess

from models.classical_model import build_model, train_model, predict

def main():
    X, y = load_data()

    df = X.copy()
    df['Outcome']=y

    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"Data preprocessed by teammate's code.")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    model = build_model()
    train_model(model, X_train, y_train)
    print("Classical model trained.")

    y_pred = predict(model, X_test)

    from sklearn.metrics import accuracy_score, classification_report
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===Classical model performance ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"\nClassification Report: ")
    print(classification_report(y_test, y_pred))

if __name__== "__main__":
    main() 