import numpy as np
from .JTCorrelator import classical_jtc


def calculate_trace_distance_diag(p: np.ndarray, q: np.ndarray) -> float:
    """Trace distance between *diagonal* density operators."""
    return 0.5 * np.abs(p - q).sum()


def calculate_fidelity_distance_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """Computes fidelity distance for full matrices via d = sqrt(1 - F²).

    Args:
        A: First density matrix
        B: Second density matrix

    Returns:
        Fidelity distance sqrt(1 - F²) where F is the JTC similarity
    """
    F = classical_jtc(A.flatten(), B.flatten(), shape=A.shape)[2] ** 2
    value_inside_sqrt = 1 - F
    if value_inside_sqrt < 0:
        value_inside_sqrt = 0  # Clamp to zero due to potential floating point issues
    return np.sqrt(value_inside_sqrt)


def calculate_trace_distance_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """Trace distance for full matrices via singular values - 0.5||A-B||₁.

    Args:
        A: First density matrix
        B: Second density matrix

    Returns:
        0.5 * ||A - B||₁ (trace norm)
    """
    diff = A - B
    # NB: for Hermitian diff, singular values == |eigenvalues|
    s = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(s)
