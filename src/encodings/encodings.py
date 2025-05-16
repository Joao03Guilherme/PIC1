import numpy as np
from typing import Tuple

# ----------------------------------------------------------------
# Encoding functions
# ----------------------------------------------------------------

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return vec / ||vec|| or vec if zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


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
    vec = np.concatenate([x / norm, [norm]], dtype=np.float32)
    norm_vec = np.linalg.norm(vec)
    return vec if norm_vec == 0 else (vec / norm_vec)

# ----------------------------------------------------------------
# Density matrix utilities
# ----------------------------------------------------------------

def compute_density_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    """Compute density matrix ρ = |ψ⟩⟨ψ| from vector ψ."""
    return np.outer(vector, vector)


def is_density_matrix(rho: np.ndarray) -> bool:
    """Check if rho is Hermitian, PSD, and trace ≈ 1."""
    if not np.allclose(rho, rho.conj().T):
        print("Matrix is not Hermitian")
        return False
    eigs = np.linalg.eigvals(rho)
    if np.any(eigs < -1e-10):
        print("Matrix is not positive semi-definite")
        return False
    if not np.isclose(np.trace(rho), 1):
        print("Trace is not 1")
        return False
    return True


def calculate_purity_from_density_matrix(rho: np.ndarray) -> float:
    """Return Tr(ρ²)."""
    return float(np.trace(rho @ rho))


def calculate_purity_from_vector(vector: np.ndarray) -> float:
    """Return purity of vector-based state: (1 + ||ψ||²) / 2."""
    return (1 + np.linalg.norm(vector) ** 2) / 2

# ----------------------------------------------------------------
# Gram matrix encodings
# ----------------------------------------------------------------

def gram_matrix_encoding(data_vector: np.ndarray) -> np.ndarray:
    """Encode vector via gram matrix normalized to trace 1."""
    v = normalize_vector(data_vector)
    gm = np.outer(v, v)
    return gm / np.trace(gm)


def gram_matrix_decoding(rho: np.ndarray) -> np.ndarray:
    """Decode vector from density matrix by principal eigenvector."""
    eigs, vecs = np.linalg.eigh(rho)
    idx = np.argmax(eigs)
    v = vecs[:, idx]
    return np.real(v) if np.all(np.isreal(v)) else v

# ----------------------------------------------------------------
# encoding_toolbox.py  –  classical-vector → density-matrix helpers
# ----------------------------------------------------------------

def _outer(psi: np.ndarray) -> np.ndarray:
    """|ψ⟩⟨ψ|  (assumes ψ normalised)."""
    return np.outer(psi, psi.conj())

def _norm(v: np.ndarray, eps: float = 1e-12) -> float:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero vector cannot be normalised.")
    return n

# ----------------------------------------------------------------
# 1.  Amplitude encoding  (pure state)
# ----------------------------------------------------------------

def density_amplitude(x: np.ndarray) -> np.ndarray:
    """
    ρ = |x_norm⟩⟨x_norm|  where |x_norm⟩ = x / ||x||₂.
    Dimension = len(x); qubit cost = ceil(log₂ N).
    """
    psi = x.astype(np.complex128) / _norm(x)
    return _outer(psi)

# ----------------------------------------------------------------
# 2.  Basis-probability encoding (classical mixture)
# ----------------------------------------------------------------

def density_basis_prob(p: np.ndarray) -> np.ndarray:
    """
    Treat p as a (non-negative) probability vector –-> diagonal density.
    ρ_ii = p_i / Σ p_i.
    """
    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative.")
    Z = p.sum()
    if Z == 0:
        raise ValueError("Probability vector all zeros.")
    return np.diag(p / Z).astype(np.complex128)

# ----------------------------------------------------------------
# 3.  Angle (rotation) encoding  (tensor product of single-qubit states)
# ----------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    """Single-qubit |0⟩ –Ry(θ)–> cos(θ/2)|0⟩ + sin(θ/2)|1⟩."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([c, s], dtype=np.complex128)

def density_angle(x: np.ndarray, scale: float = np.pi) -> np.ndarray:
    """
    Map feature x_j ∈ ℝ to θ_j = scale * x_j.
    Final state = ⊗_j Ry(θ_j)|0⟩.  (N qubits)
    Density matrix returned is 2^N × 2^N.
    """
    states = [_ry(scale * xi) for xi in x]
    psi = states[0]
    for s in states[1:]:
        psi = np.kron(psi, s)
    return _outer(psi)

# ----------------------------------------------------------------
# 4.  Phase encoding  (|+⟩ –Rz→ state)  – also N qubits
# ----------------------------------------------------------------

def _rz(phi: float) -> np.ndarray:
    return np.array([1.0, np.exp(1j * phi)], dtype=np.complex128) / np.sqrt(2)

def density_phase(x: np.ndarray, scale: float = 2 * np.pi) -> np.ndarray:
    """ρ from ⊗_j Rz(scale·x_j)|+⟩."""
    states = [_rz(scale * xi) for xi in x]
    psi = states[0]
    for s in states[1:]:
        psi = np.kron(psi, s)
    return _outer(psi)

# ----------------------------------------------------------------
# 5.  Inverse stereographic projection → one-qubit Bloch state
# ----------------------------------------------------------------

def density_inv_stereo(v: np.ndarray) -> np.ndarray:
    """
    Map ℝ² (or ℝ³ with last dim ignored) to Bloch sphere via inverse
    stereographic projection.  Return 2×2 density matrix.
    """
    if v.size < 2:
        raise ValueError("Need at least two coordinates for stereographic map.")
    x, y = float(v[0]), float(v[1])
    r2 = x * x + y * y
    theta = 2 * np.arctan(np.sqrt(r2))
    phi = np.arctan2(y, x)
    # Bloch: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    psi = np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
        dtype=np.complex128,
    )
    return _outer(psi)

# ----------------------------------------------------------------
# 6.  “Informative” (phase × amplitude) encoding  – used in QI kernels
# ----------------------------------------------------------------

def density_informative(x: np.ndarray) -> np.ndarray:
    """
    Encode  x_j  into both amplitude AND phase:
       ψ_j = e^{i π x_j} · x_j / ||x||₂    (j=0..N-1)
    Gives a richer inner-product kernel than amplitude alone.
    """
    amp = x.astype(np.float64) / _norm(x)
    phase = np.exp(1j * np.pi * x.astype(np.float64))
    psi = (amp * phase).astype(np.complex128)
    return _outer(psi)
