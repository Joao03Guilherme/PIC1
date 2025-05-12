# src/models/QuantumNearestMean/quantum_nearestmean_network.py
import numpy as np
from typing import Literal, Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances

# reuse the utility that already builds custom metrics for you
from ..utils import make_distance_fn


class QuantumNearestMeanClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum-inspired Nearest Mean Classifier (QNMC).

    Parameters
    ----------
    encoding : {'diag_prob', 'stereographic', 'informative'}, default='diag_prob'
        How to map an input vector x ∈ ℝ^d to a density operator ρ_x.
        * diag_prob      : ρ_x = diag(x / sum(x))               (rank-≥1, diagonal)
        * stereographic  : ρ_x = |ψ⟩⟨ψ| with ψ given by Eq.(2)–(3) of Sergioli et al.
        * informative    : ρ_x = |ψ⟩⟨ψ| with ψ given by Eq.(7)–(10) of Sergioli et al.

    distance : {'trace', <any string accepted by make_distance_fn>}, default='trace'
        Distance between density operators.  If not 'trace', the string is passed
        to ``make_distance_fn`` and will be applied to *vectors* that represent
        density matrices (diagonals for 'diag_prob'; flattened matrices otherwise).

    distance_squared : bool, default=False
        Forwarded to ``make_distance_fn`` for your JTC-based metrics.

    """

    def __init__(
        self,
        encoding: Literal["diag_prob", "stereographic", "informative"] = "stereographic",
        distance: str = "trace",
        distance_squared: bool = False,
        random_state: Optional[int] = None,
    ):
        self.encoding = encoding
        self.distance = distance
        self.distance_squared = distance_squared
        self.random_state = random_state

    # ------------------------------------------------------------------
    #                      ENCODING HELPERS
    # ------------------------------------------------------------------
    def _encode_diag_prob(self, x: np.ndarray) -> np.ndarray:
        """Diagonal-probability encoding: returns the *vector* of diagonal entries."""
        x = x.astype(np.float32, copy=False)
        s = x.sum()
        if s == 0.0:           # blank image -> uniform distribution
            return np.full_like(x, 1.0 / x.size, dtype=np.float32)
        return x / s

    def _encode_stereographic(self, x: np.ndarray) -> np.ndarray:
        """Return ψ ∈ ℝ^{d+1} (unit vector) – Eq.(2)–(3) in the paper."""
        norm2 = np.dot(x, x)
        psi = np.concatenate([2 * x, [norm2 - 1]], dtype=np.float32)
        psi /= (norm2 + 1)          # normalisation factor
        return psi / np.linalg.norm(psi)

    def _encode_informative(self, x: np.ndarray) -> np.ndarray:
        """Return ψ ∈ ℝ^{d+1} – Eq.(7)–(10) in the paper."""
        x = x.astype(np.float32, copy=False)
        norm = np.linalg.norm(x)
        if norm == 0.0:
            psi = np.zeros(x.size + 1, dtype=np.float32)
            psi[-1] = 1.0
            return psi
        vec = np.concatenate([x / norm, [norm]], dtype=np.float32)
        vec /= np.sqrt(norm**2 + 1.0)
        return vec

    # choose encoder at runtime
    def _encode(self, x: np.ndarray) -> np.ndarray:
        if self.encoding == "diag_prob":
            return self._encode_diag_prob(x)
        elif self.encoding == "stereographic":
            return self._encode_stereographic(x)
        elif self.encoding == "informative":
            return self._encode_informative(x)
        else:
            raise ValueError(f"Unknown encoding '{self.encoding}'")

    # ------------------------------------------------------------------
    #                      DISTANCE HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _trace_distance_diag(p: np.ndarray, q: np.ndarray) -> float:
        """Trace distance between *diagonal* density operators."""
        return 0.5 * np.abs(p - q).sum()

    @staticmethod
    def _trace_distance_matrix(A: np.ndarray, B: np.ndarray) -> float:
        """Trace distance for full matrices via singular values - 0.5||A-B||₁."""
        diff = A - B
        # NB: for Hermitian diff, singular values == |eigenvalues|
        s = np.linalg.svd(diff, compute_uv=False)
        return 0.5 * s.sum()

    # user-supplied or JTC distance (vector form)
    def _make_vector_metric(self) -> Callable[[np.ndarray, np.ndarray], float]:
        return make_distance_fn(name=self.distance, squared=self.distance_squared)

    # ------------------------------------------------------------------
    #                       FIT
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute quantum centroids (average density operators) for each class.
        """
        X = X.astype(np.float32, copy=False)
        self.classes_ = np.unique(y)

        # containers for accumulating sums
        if self.encoding == "diag_prob":
            sums = {label: np.zeros(X.shape[1], dtype=np.float32) for label in self.classes_}
        else:
            dim = X.shape[1] + 1 if self.encoding != "diag_prob" else X.shape[1]
            sums = {label: np.zeros((dim, dim), dtype=np.float32) for label in self.classes_}

        counts = {label: 0 for label in self.classes_}

        # accumulate outer products (or diagonals)
        for xi, label in zip(X, y):
            enc = self._encode(xi)
            if self.encoding == "diag_prob":
                sums[label] += enc                 # vector
            else:
                sums[label] += np.outer(enc, enc)  # full density matrix
            counts[label] += 1

        # final centroids
        if self.encoding == "diag_prob":
            self.centroids_ = {
                label: sums[label] / counts[label] for label in self.classes_
            }
        else:
            self.centroids_ = {
                label: sums[label] / counts[label] for label in self.classes_
            }

        # pick appropriate distance function
        if self.distance == "trace":
            if self.encoding == "diag_prob":
                self._metric_ = self._trace_distance_diag
            else:
                self._metric_ = self._trace_distance_matrix
        else:
            # any custom metric works on *vectors*; choose representation:
            vecmetric = self._make_vector_metric()

            if self.encoding == "diag_prob":
                self._metric_ = vecmetric
            else:
                # flatten the symmetric matrix ; store upper-triangular part for speed
                def _mat_to_vec(M: np.ndarray):
                    return M[np.triu_indices_from(M)]
                self._metric_ = lambda A, B: vecmetric(_mat_to_vec(A), _mat_to_vec(B))

        return self

    # ------------------------------------------------------------------
    #                       PREDICT
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Classifier has not been fitted.")

        X = X.astype(np.float32, copy=False)
        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)

        for idx, xi in enumerate(X):
            enc = self._encode(xi)

            # choose representation to feed to distance metric
            if self.distance == "trace" or self.encoding != "diag_prob":
                rep_x = enc if self.encoding == "diag_prob" else np.outer(enc, enc)

            best_dist = np.inf
            best_lbl = None

            for lbl in self.classes_:
                if self.encoding == "diag_prob":
                    dist = self._metric_(enc, self.centroids_[lbl])
                else:
                    dist = self._metric_(rep_x, self.centroids_[lbl])

                if dist < best_dist:
                    best_dist, best_lbl = dist, lbl

            preds[idx] = best_lbl

        return preds
