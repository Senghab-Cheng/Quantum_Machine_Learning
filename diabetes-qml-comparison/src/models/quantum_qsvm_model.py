import sys
from pathlib import Path

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config


def build_qsvm():
    """Build a Quantum Support Vector Machine model."""
    feature_map = zz_feature_map(
        feature_dimension=config.N_QUBITS,
        reps=config.QSVM_FEATURE_MAP_REPS,
        entanglement=config.QSVM_ENTANGLEMENT,
    )

    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    return QSVC(quantum_kernel=quantum_kernel, C=config.QSVM_C)


def train_qsvm(model, X_train, y_train):
    """Train the QSVM model."""
    model.fit(X_train, y_train)
    return model


def predict_qsvm(model, X_test):
    """Predict using the QSVM model."""
    return model.predict(X_test)
