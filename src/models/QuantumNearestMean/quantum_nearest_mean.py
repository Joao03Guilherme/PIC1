import numpy as np
from functools import reduce
from typing import Literal, Callable, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import defaultdict

# Re‑use the utility that already builds custom metrics for you
from ..utils import make_distance_fn

from ...distance.JTCorrelator import classical_jtc
# from ...distance.OpticalJTCorrelator import OpticalJTCorrelator
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
    encode_length_scaled,
    normalize_vector,
)


class QuantumNearestMeanClassifier(BaseEstimator, ClassifierMixin):
    """Quantum‑inspired Nearest‑Mean Classifier (QNMC) with support for *m* tensor‑product copies.

    Parameters
    ----------
    encoding : {'diag_prob', 'stereographic', 'informative', 'standard', 'length_scaled'} (default='stereographic')
        How to map an input vector *x* ∈ R^d to a quantum state ρₓ.

    distance : {'trace', 'fidelity', 'optical_classical_jtc', ...} (default='fidelity')
        Distance between density operators (or their vector representations).

    copies : int, default=1
        Number *m* of identical copies ρ⊗m used to enlarge the Hilbert space.  *copies=1* recovers the
        original classifier.  Beware: dimension grows as d^m (diag_prob) or d^{2m} (matrix encodings).

    distance_squared : bool, default=False
        Forwarded to :func:`make_distance_fn` for JTC‑based metrics.

    optical_correlator : Optional[OpticalJTCorrelator]
        Required when *distance=='optical_classical_jtc'*.
    """

    # ---------------------------------------------------------------------
    #                            INIT
    # ---------------------------------------------------------------------
    def __init__(
        self,
        encoding: Literal[
            "diag_prob", "stereographic", "informative", "standard", "length_scaled"
        ] = "stereographic",
        distance: str = "fidelity",
        copies: int = 1,
        distance_squared: bool = False,
        optical_correlator = None,
        random_state: Optional[int] = None,
    ):
        if copies < 1 or not isinstance(copies, int):
            raise ValueError("copies must be a positive integer")

        self.encoding = encoding
        self.distance = distance
        self.copies = copies
        self.distance_squared = distance_squared
        self.optical_correlator = optical_correlator
        self.random_state = random_state

        # Ensure optical correlator is provided when required
        if distance == "optical_classical_jtc" and optical_correlator is None:
            raise ValueError(
                "optical_correlator must be provided for 'optical_classical_jtc' distance"
            )

    # ---------------------------------------------------------------------
    #                       ENCODING HELPERS
    # ---------------------------------------------------------------------
    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Map a raw feature vector to its *single‑copy* quantum representation."""
        if self.encoding == "diag_prob":
            return encode_diag_prob(x)
        elif self.encoding == "stereographic":
            return encode_stereographic(x)
        elif self.encoding == "informative":
            return encode_informative(x)
        elif self.encoding == "standard":
            return normalize_vector(x)
        elif self.encoding == "length_scaled":
            return encode_length_scaled(x)
        else:
            raise ValueError(f"Unknown encoding '{self.encoding}'")

    # ---------------------------------------------------------------------
    #                    TENSOR‑PRODUCT  HELPERS
    # ---------------------------------------------------------------------
    def _tensor_product(self, obj: np.ndarray, diag: bool) -> np.ndarray:
        """Return *m*-fold tensor product of *obj*.

        * If *diag* is True, *obj* is treated as a probability vector and the
          function returns the Kronecker product of vectors.
        * Otherwise *obj* is a density **matrix** and the Kronecker product is
          applied to the matrix (⨂ along both axes).
        """
        if self.copies == 1:
            return obj

        # numpy.kron already performs left Kronecker product for vectors/matrices
        return reduce(np.kron, [obj] * self.copies)

    # ---------------------------------------------------------------------
    #                       DISTANCE  HELPERS
    # ---------------------------------------------------------------------
    def _make_vector_metric(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """Wrap :func:`make_distance_fn` to account for vector‑shaped inputs."""
        n_features = getattr(self, "n_features_", None)
        shape = None
        if n_features is not None and n_features > 0:
            # Attempt to build a near‑square shape for JTC correlator visualisation
            h = int(np.sqrt(n_features))
            while h > 1 and n_features % h != 0:
                h -= 1
            shape = (h, n_features // h) if h > 1 else (1, n_features)
        else:
            shape = (28, 28)  # sensible default for image datasets

        return make_distance_fn(
            name=self.distance,
            squared=self.distance_squared,
            shape=shape,
            optical_correlator=self.optical_correlator,
        )

    # ---------------------------------------------------------------------
    #                                FIT
    # ---------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Compute class‑wise quantum centroids."""
        X = X.astype(np.float32, copy=False)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        # Determine dimensionality after encoding and tensor product
        if self.encoding == "diag_prob":
            vec_dim = self.n_features_ ** self.copies
            sums = {lbl: np.zeros(vec_dim, dtype=np.float32) for lbl in self.classes_}
        else:
            # Dimension of a single‑copy matrix
            if self.encoding in ("stereographic", "informative", "length_scaled"):
                d_single = X.shape[1] + 1
            else:  # standard
                d_single = X.shape[1]
            d_tensor = d_single ** self.copies
            sums = {
                lbl: np.zeros((d_tensor, d_tensor), dtype=np.float32)
                for lbl in self.classes_
            }

        counts = defaultdict(int)

        # -----------------------------------------------------------------
        #                    ACCUMULATE CLASS CENTROIDS
        # -----------------------------------------------------------------
        for xi, lbl in zip(X, y):
            enc = self._encode(xi)  # vector form |ψ⟩  or prob‐vector

            if self.encoding == "diag_prob":
                enc_tp = self._tensor_product(enc, diag=True)  # probability vector
                sums[lbl] += enc_tp
            else:
                rho = np.outer(enc, enc)  # single‑copy density matrix
                rho_tp = self._tensor_product(rho, diag=False)
                sums[lbl] += rho_tp

            counts[lbl] += 1

        # Normalise sums to obtain centroids
        self.centroids_ = {lbl: sums[lbl] / counts[lbl] for lbl in self.classes_}

        # -----------------------------------------------------------------
        #                       SELECT DISTANCE FUNCTION
        # -----------------------------------------------------------------
        if self.distance == "trace":
            if self.encoding == "diag_prob":
                self._metric_ = calculate_trace_distance_diag
            else:
                self._metric_ = calculate_trace_distance_matrix
        elif self.distance == "fidelity":
            if self.encoding == "diag_prob":
                self._metric_ = lambda p, q: 1.0 - np.sum(np.sqrt(p * q))
            else:
                self._metric_ = calculate_fidelity_distance_matrix
        else:  # JTC or other custom metric working on vectors
            vecmetric = self._make_vector_metric()

            if self.encoding == "diag_prob":
                self._metric_ = vecmetric
            else:
                # Need a wrapper converting matrices to flattened vectors
                def _mat_to_vec(M: np.ndarray):
                    return M.flatten()

                def custom_metric(A, B):
                    vA, vB = _mat_to_vec(A), _mat_to_vec(B)
                    return vecmetric(vA, vB)

                self._metric_ = custom_metric

        return self

    # ---------------------------------------------------------------------
    #                              PREDICT
    # ---------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "centroids_"):
            raise RuntimeError("Classifier has not been fitted.")

        X = X.astype(np.float32, copy=False)
        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)

        for i, xi in enumerate(X):
            enc = self._encode(xi)

            if self.encoding == "diag_prob":
                rep_x = self._tensor_product(enc, diag=True)
            else:
                rho = np.outer(enc, enc)
                rep_x = self._tensor_product(rho, diag=False)

            best_dist, best_lbl = np.inf, None
            for lbl in self.classes_:
                dist = self._metric_(rep_x, self.centroids_[lbl])
                if dist < best_dist:
                    best_dist, best_lbl = dist, lbl
            preds[i] = best_lbl

        return preds
