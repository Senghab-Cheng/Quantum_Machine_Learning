from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Check where optimizers are
try:
    from qiskit_algorithms.optimizers import COBYLA
    print("COBYLA imported from qiskit_algorithms.optimizers")
except ImportError:
    try:
        from qiskit.algorithms.optimizers import COBYLA
        print("COBYLA imported from qiskit.algorithms.optimizers")
    except ImportError:
        print("Could not import COBYLA")

print("Imports checked.")
