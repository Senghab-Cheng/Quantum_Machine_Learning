import sys
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from inference import fit_pipeline
from models.classical_models import get_all_models, train_all_models, evaluate_all_models


def run_verification():
    print("Starting verification of all models (including Quantum)...")

    print("Loading and fitting preprocessing pipeline...")
    pipeline = fit_pipeline()
    X_train = pipeline["X_train"]
    y_train = pipeline["y_train"]
    X_test = pipeline["X_test"]
    y_test = pipeline["y_test"]

    # Use a small subset for quick verification
    subset_size = 30
    X_subset = X_train[:subset_size]
    y_subset = y_train[:subset_size]
    X_test_subset = X_test[:min(subset_size // 2, len(X_test))]
    y_test_subset = y_test[:min(subset_size // 2, len(y_test))]

    print(f"Train subset: {X_subset.shape}, Test subset: {X_test_subset.shape}")
    print(f"Feature dimensions: {X_subset.shape[1]} (expected {4} for quantum models)")

    print("Initializing models...")
    all_models = get_all_models()
    print(f"Models to train: {list(all_models.keys())}")

    print("Training models (quantum models may take a moment)...")
    trained_models, training_info = train_all_models(all_models, X_subset, y_subset)

    print("Evaluating models...")
    results = evaluate_all_models(
        trained_models, X_test_subset, y_test_subset, training_info
    )

    print("\nEvaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n--- {model_name} ---")
        for metric, value in metrics.items():
            if metric == "Confusion Matrix":
                print(f"{metric}:\n{value}")
            elif isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")


if __name__ == "__main__":
    try:
        run_verification()
        print("\nVerification successful.")
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        raise
