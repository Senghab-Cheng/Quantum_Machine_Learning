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
CLASSICAL_SOLVER = "lbfgs"
CLASSICAL_CLASS_WEIGHT = "balanced"

# Decision Tree hyperparameters
DT_MAX_DEPTH = 5
DT_MIN_SAMPLES_SPLIT = 10
DT_MIN_SAMPLES_LEAF = 1
DT_CRITERION = "gini"
DT_RANDOM_STATE = RANDOM_STATE

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_SPLIT = 2
RF_MAX_FEATURES = "sqrt"
RF_RANDOM_STATE = RANDOM_STATE

# Hyperparameter tuning
CV_FOLDS = 5
TUNING_SCORING = "accuracy"

# Evaluation & logging
RANDOM_SEED = RANDOM_STATE
RESULTS_PATH = PROJECT_ROOT / "results"
MODELS_PATH = PROJECT_ROOT / "models"
LOGS_PATH = PROJECT_ROOT / "logs"
CLASSICAL_MODELS_PATH = RESULTS_PATH / "classical_models"
CONFUSION_MATRICES_PATH = RESULTS_PATH / "confusion_matrices"

# Ensure output directories exist
RESULTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)
CLASSICAL_MODELS_PATH.mkdir(exist_ok=True)
CONFUSION_MATRICES_PATH.mkdir(exist_ok=True)