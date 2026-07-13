from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"

# Global dataset parameters
N_QUBITS = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# VQC (Variational Quantum Classifier) hyperparameters
VQC_MAX_ITER = 100
VQC_LEARNING_RATE = 0.01
VQC_OPTIMIZER = "COBYLA"

# QSVM (Quantum Support Vector Machine) hyperparameters
QSVM_FEATURE_MAP_REPS = 2
QSVM_ENTANGLEMENT = "linear"
QSVM_C = 1.0

# Classical model hyperparameters
CLASSICAL_MAX_ITER = 1000
CLASSICAL_C = 1.0
CLASSICAL_RANDOM_STATE = RANDOM_STATE

# Evaluation & logging
RANDOM_SEED = RANDOM_STATE
RESULTS_PATH = PROJECT_ROOT / "results"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"

# Ensure output directories exist
RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)