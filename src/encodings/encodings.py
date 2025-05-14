import numpy as np


def normalize_vector(vec):
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
    """Diagonal-probability encoding: returns the *vector* of diagonal entries."""
    x = x.astype(np.float32, copy=False)
    s = x.sum()
    if s == 0.0:  # blank image -> uniform distribution
        return np.full_like(x, 1.0 / x.size, dtype=np.float32)
    return x / s


def encode_stereographic(x: np.ndarray) -> np.ndarray:
    """Return ψ ∈ ℝ^{d+1} (unit vector) – Eq.(2)–(3) in the paper."""
    norm2 = np.dot(x, x)
    psi = np.concatenate([2 * x, [norm2 - 1]], dtype=np.float32)
    psi /= norm2 + 1  # normalisation factor
    # Ensure psi is a unit vector, handling potential norm of 0
    psi_norm = np.linalg.norm(psi)
    if psi_norm == 0:
        return psi
    return psi / psi_norm


def encode_informative(x: np.ndarray) -> np.ndarray:
    """Return ψ ∈ ℝ^{d+1} – Eq.(7)–(10) in the paper."""
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x)
    if norm == 0.0:
        psi = np.zeros(x.size + 1, dtype=np.float32)
        psi[-1] = 1.0
        return psi
    vec = np.concatenate([x / norm, [norm]], dtype=np.float32)
    # Ensure vec is a unit vector
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:  # Should not happen if norm > 0 or norm == 0 handled above
        return vec
    return vec / vec_norm
