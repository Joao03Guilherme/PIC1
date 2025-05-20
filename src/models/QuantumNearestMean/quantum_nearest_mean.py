# src/models/QuantumNearestMean/quantum_nearestmean_network.py
import numpy as np
from typing import Literal, Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin

# reuse the utility that already builds custom metrics for you
from ..utils import make_distance_fn

from ...distance.JTCorrelator import classical_jtc
from ...distance.OpticalJTCorrelator import OpticalJTCorrelator
from ...distance.quantum_distances import (
    calculate_trace_distance_diag,
    calculate_fidelity_distance_matrix,
    calculate_trace_distance_matrix,
)

# Import encoding functions
from ...encodings.encodings import (
    encode_diag_prob,
    encode_stereographic,
    encode_informative,
    normalize_vector,
)


class QuantumNearestMeanClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum-inspired Nearest Mean Classifier (QNMC).

    Parameters
    ----------
    encoding : {'diag_prob', 'stereographic', 'informative'}, default='diag_prob'
        How to map an input vector x ∈ R^d to a density operator p_x.
        * diag_prob      : p_x = diag(x / sum(x))               (rank-≥1, diagonal)
        * stereographic  : p_x = |ψ⟩⟨ψ| with ψ given by Eq.(2)-(3) of Sergioli et al.
        * informative    : p_x = |ψ⟩⟨ψ| with ψ given by Eq.(7)-(10) of Sergioli et al.

    distance : {'trace', 'fidelity', 'optical_classical_jtc'}, default='trace'
        Distance between density operators.  If not 'trace' or 'fidelity',
        the string is passed to `make_distance_fn` and will be applied to
        *vectors* that represent density matrices (diagonals for 'diag_prob';
        flattened matrices otherwise).
        'optical_classical_jtc' uses the OpticalJTCorrelator.

    distance_squared : bool, default=False
        Forwarded to `make_distance_fn` for your JTC-based metrics.

    optical_correlator : OpticalJTCorrelator, optional
        An instance of OpticalJTCorrelator, required if distance is 'optical_classical_jtc'.

    """

    def __init__(
        self,
        encoding: Literal[
            "diag_prob", "stereographic", "informative", "standard"
        ] = "stereographic",
        distance: str = "fidelity",
        distance_squared: bool = False,
        optical_correlator: Optional[OpticalJTCorrelator] = None,
        random_state: Optional[int] = None,
    ):
        self.encoding = encoding
        self.distance = distance
        self.distance_squared = distance_squared
        self.optical_correlator = optical_correlator
        self.random_state = random_state

        # Validate optical_correlator is provided if using optical distance
        if distance == "optical_classical_jtc" and optical_correlator is None:
            raise ValueError(
                "optical_correlator must be provided for 'optical_classical_jtc' distance"
            )

    # ------------------------------------------------------------------
    #                      ENCODING HELPERS
    # ------------------------------------------------------------------
    # Encoding methods are now imported from src.encodings.encodings

    # choose encoder at runtime
    def _encode(self, x: np.ndarray) -> np.ndarray:
        if self.encoding == "diag_prob":
            return encode_diag_prob(x)
        elif self.encoding == "stereographic":
            return encode_stereographic(x)
        elif self.encoding == "informative":
            return encode_informative(x)
        elif self.encoding == "standard":
            return normalize_vector(x)
        else:
            raise ValueError(f"Unknown encoding '{self.encoding}'")

    # ------------------------------------------------------------------
    #                      DISTANCE HELPERS
    # ------------------------------------------------------------------
    # Distance helper methods are now imported from src.distance.quantum_distances

    # user-supplied or JTC distance (vector form)
    def _make_vector_metric(self) -> Callable[[np.ndarray, np.ndarray], float]:
        # Include image shape and optical correlator for optical_classical_jtc distance
        n_features = getattr(self, "n_features_", None)
        shape = None  # Default shape
        if n_features is not None:
            # Determine H, W for reshaping vectors to images
            h_candidate = int(np.sqrt(n_features))
            if n_features > 0:  # Ensure n_features is positive
                while h_candidate > 0 and n_features % h_candidate != 0:
                    h_candidate -= 1

            if (
                h_candidate > 0 and n_features % h_candidate == 0
            ):  # Found a valid factor
                H = h_candidate
                W = n_features // H
            elif (
                n_features > 0
            ):  # if n_features is prime or no integer factor found, use 1 x n_features
                H = 1
                W = n_features
            else:  # Fallback for n_features = 0 or other unexpected cases
                H, W = 0, 0
                print(
                    f"Warning: Could not determine H, W for n_features={n_features}. Using ({H},{W})."
                )
            shape = (H, W)
        else:
            # Default shape if n_features is not yet set (e.g. MNIST default)
            shape = (28, 28)
            print(
                f"Warning: n_features_ not set in _make_vector_metric. Defaulting shape to {shape}."
            )

        return make_distance_fn(
            name=self.distance,
            squared=self.distance_squared,
            shape=shape,
            optical_correlator=self.optical_correlator,
        )

    # ------------------------------------------------------------------
    #                       FIT
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute quantum centroids (average density operators) for each class.
        """
        X = X.astype(np.float32, copy=False)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]  # Store for reshaping in distance calculation

        if self.encoding == "diag_prob":
            sums = {
                label: np.zeros(X.shape[1], dtype=np.float32) for label in self.classes_
            }

        elif self.encoding in ("stereographic", "informative"):  # d  ->  d+1
            dim = X.shape[1] + 1
            sums = {
                label: np.zeros((dim, dim), dtype=np.float32) for label in self.classes_
            }

        elif self.encoding == "standard":  # stays at d
            dim = X.shape[1]
            sums = {
                label: np.zeros((dim, dim), dtype=np.float32) for label in self.classes_
            }
        else:
            raise ValueError("No valid encoding defined")

        counts = {label: 0 for label in self.classes_}

        # accumulate outer products (or diagonals)
        for xi, label in zip(X, y):
            enc = self._encode(xi)
            if self.encoding == "diag_prob":
                sums[label] += enc  # vector
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
                self._metric_ = calculate_trace_distance_diag
            else:
                self._metric_ = calculate_trace_distance_matrix

        elif self.distance == "fidelity":
            if self.encoding == "diag_prob":
                self._metric_ = lambda p, q: 1.0 - np.sum(
                    np.sqrt(p * q)
                )  # Corrected fidelity for probability vectors
            else:
                self._metric_ = calculate_fidelity_distance_matrix

        # All JTC-based distances (classical, optical, other custom) are handled here
        else:
            vecmetric = self._make_vector_metric()

            if self.encoding == "diag_prob":
                self._metric_ = vecmetric
            else:
                # For non-diag_prob encodings, centroids are density matrices.
                # These need to be converted to vectors to be used with vecmetric.
                # The current _mat_to_vec flattens the upper triangular part.
                # This may not be appropriate for JTC methods expecting image-like vectors
                # of shape derived from self.n_features_.
                # This behavior is now consistent for all JTC-like metrics obtained via make_distance_fn.
                def _mat_to_vec(M: np.ndarray):
                    # Ensure M is 2D
                    if M.ndim == 1:
                        # This case might occur if encoding produces a vector but isn't 'diag_prob'.
                        # Or if a 1D feature vector was encoded to a 1D 'density matrix' (unlikely for stereographic/informative).
                        # For now, assume if not 'diag_prob', M should be a matrix.
                        # If it's already a vector, it might be an error or needs specific handling.
                        # Returning it as is, but this could be problematic if its length doesn't match expectations.
                        print(
                            f"Warning: _mat_to_vec received a 1D array for non-diag_prob encoding. Length: {len(M)}"
                        )
                        return M
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
            if self.encoding == "diag_prob":
                # no rep_x needed, enc is the vector
                pass
            else:
                # rep_x is the density matrix
                rep_x = np.outer(enc, enc)

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
