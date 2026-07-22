import sys
from pathlib import Path

from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC

try:
    from qiskit.algorithms.optimizers import COBYLA
except ImportError:
    from qiskit_algorithms.optimizers import COBYLA

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _path in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config


def build_vqc():
    """Build a Variational Quantum Classifier model."""
    feature_map = zz_feature_map(
        feature_dimension=config.N_QUBITS,
        reps=1,
    )
    ansatz = real_amplitudes(
        num_qubits=config.N_QUBITS,
        reps=config.QSVM_FEATURE_MAP_REPS,
    )
    optimizer = COBYLA(maxiter=config.VQC_MAX_ITER)
    sampler = StatevectorSampler()

    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
    )


def train_vqc(model, X_train, y_train):
    """Train the VQC model."""
    model.fit(X_train, y_train)
    return model


def predict_vqc(model, X_test):
    """Predict using the VQC model."""
    return model.predict(X_test)
