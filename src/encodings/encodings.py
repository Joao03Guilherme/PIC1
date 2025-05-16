import numpy as np
from typing import Tuple

# ----------------------------------------------------------------
# Encoding functions
# ----------------------------------------------------------------

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return vec / ||vec|| or vec if zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def is_density_matrix(rho):
    if not np.allclose(rho, np.conj(rho.T)):
        print("Matrix is not Hermitian")
        return False

    eigenvalues = np.linalg.eigvals(rho)
    if np.any(eigenvalues < -1e-10):  # allow small numerical errors
        print("Matrix is not positive semi-definite")
        return False

    if not np.isclose(np.trace(rho), 1):
        print("Trace is not 1")
        return False

    return True


def calculate_purity_from_density_matrix(rho):
    return np.trace(rho @ rho)


def calculate_purity_from_vector(vector):
    return (1 + np.linalg.norm(vector) ** 2) / 2


def gram_matrix_encoding(data_vector):
    data_vector = normalize_vector(data_vector)
    gram_matrix = np.dot(data_vector, data_vector.T)
    rho = gram_matrix / np.trace(gram_matrix)
    return rho


def gram_matrix_decoding(rho):
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    # Find the eigenvector with the largest eigenvalue
    idx = np.argmax(eigenvalues)
    decoded_vector = eigenvectors[:, idx]
    # Ensure the vector is real if the original was real
    if np.all(np.isreal(decoded_vector)):
        decoded_vector = np.real(decoded_vector)
    # Return as column vector
    return decoded_vector.reshape(-1, 1)


def compute_density_matrix_from_vector(vector):
    """Compute the density matrix from a vector."""
    rho = np.outer(vector, vector)
    return rho

def encode_diag_prob(x: np.ndarray) -> np.ndarray:
    """Diagonal probability encoding: p_x = x / sum(x)."""
    x = x.astype(np.float32, copy=False)
    total = x.sum()
    if total == 0.0:
        return np.full_like(x, 1.0 / x.size, dtype=np.float32)
    return (x / total).astype(np.float32)


def encode_stereographic(x: np.ndarray) -> np.ndarray:
    """Return stereographic encoding ψ ∈ ℝ^(d+1), normalized."""
    x = x.astype(np.float32, copy=False)
    norm2 = np.dot(x, x)
    vec = np.concatenate([2 * x, [norm2 - 1]], dtype=np.float32)
    vec /= norm2 + 1
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else (vec / norm)


def encode_informative(x: np.ndarray) -> np.ndarray:
    """Informative encoding: amplitude × phase mapping."""
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x)
    if norm == 0.0:
        psi = np.zeros(x.size + 1, dtype=np.float32)
        psi[-1] = 1.0
        return psi
    
    vec = np.concatenate([x / norm, [1]], dtype=np.float32)
    return 1/np.sqrt(2) * vec
