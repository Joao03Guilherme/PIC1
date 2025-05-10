from __future__ import annotations

from typing import Union

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin


from ...distance.JTCorrelator import (
    phase_corr_similarity,
    classical_jtc,
)


def make_distance_fn(name: str = "phase", *, squared: bool = False):
    """Return a callable f(X, Y) compatible with sklearn.pairwise_distances."""
    if name == "phase":

        def _d(X, Y):
            d, _, _, _ = phase_corr_similarity(X, Y, shape=(28, 28))
            return d

    elif name == "classical_jtc":

        def _d(X, Y):
            d, _, _, _ = classical_jtc(
                X.reshape(28, 28),
                Y.reshape(28, 28),
                shape=(28, 28),
            )
            return d

    elif name == "euclidean":
        return "euclidean" if not squared else "sqeuclidean"
    else:
        raise ValueError(f"Unknown distance name '{name}'")

    return (lambda X, Y, *, _f=_d: _f(X, Y) ** 2) if squared else _d


# -----------------------------------------------------------------------------
# RBF network class
# -----------------------------------------------------------------------------
class RBFNet(BaseEstimator, ClassifierMixin):
    """Gaussian-RBF or correlation-based kernel network."""

    def __init__(
        self,
        n_centers: Union[int, None] = None,
        k_sigma: float = 1.0,
        n_classes: int = 10,
        use_perceptron: bool = False,
        max_iter_perceptron: int = 10,
        lr: float = 0.05,
        lambda_ridge: float = 1e-3,
        min_sigma: float = 1e-2,
        distance_name: str = "phase",
        distance_squared: bool = True,
        random_state: int | None = None,
    ) -> None:
        # Parameters must be assigned to attributes with *the same names*
        # so that scikit‑learn's clone() can find them during CV.
        self.n_centers = n_centers
        self.k_sigma = k_sigma
        self.n_classes = n_classes
        self.use_perceptron = use_perceptron
        self.max_iter_perceptron = max_iter_perceptron  # ← added
        self.lr = lr
        self.lambda_ridge = lambda_ridge  # ← added
        self.min_sigma = min_sigma
        self.distance_name = distance_name
        self.distance_squared = distance_squared
        self.random_state = random_state

        # Internal aliases (shorter names):
        self.max_iter_p = max_iter_perceptron
        self.l2 = lambda_ridge

    # --------------------------- training -----------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X_raw = X.astype(np.float32, copy=False)

        # 1. choose centres -------------------------------------------------
        if self.n_centers is None:
            self.centers_ = np.unique(X_raw, axis=0)
        else:
            self.centers_ = (
                MiniBatchKMeans(
                    self.n_centers,
                    batch_size=4096,
                    max_iter=100,
                    n_init="auto",
                    random_state=self.random_state,
                )
                .fit(X_raw)
                .cluster_centers_.astype(np.float32)
            )

        dist_fn = make_distance_fn(self.distance_name, squared=self.distance_squared)

        # 2. compute distances -------------------------------------------
        d2 = pairwise_distances(self.centers_, metric=dist_fn, n_jobs=-1)
        np.fill_diagonal(d2, np.inf)
        nn2 = d2.min(axis=1)
        sigma2 = np.maximum((self.k_sigma**2) * nn2, self.min_sigma**2)
        self.gammas_ = 0.5 / sigma2

        # 3. design matrix -----------------------------------------------
        phi = np.exp(
            -pairwise_distances(X_raw, self.centers_, metric=dist_fn, n_jobs=-1)
            * self.gammas_[None, :]
        )

        # 4. output layer -------------------------------------------------
        if self.use_perceptron:
            self._train_perceptron(phi, y)
        else:
            self._train_ridge(phi, y)
        return self

    # --------------------------- predictors ------------------------------
    def _rbf_layer(self, X: np.ndarray) -> np.ndarray:
        dist_fn = make_distance_fn(self.distance_name, squared=self.distance_squared)
        return np.exp(
            -pairwise_distances(
                X.astype(np.float32, copy=False),
                self.centers_,
                metric=dist_fn,
                n_jobs=-1,
            )
            * self.gammas_[None, :]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        phi = self._rbf_layer(X)
        return (phi @ self.W_.T).argmax(1)

    # ------------------- output-layer helpers ---------------------------
    def _train_perceptron(self, phi: np.ndarray, y: np.ndarray) -> None:
        m = phi.shape[1]
        self.W_ = np.zeros((self.n_classes, m), dtype=np.float32)
        self.W_[y, np.arange(len(y)) % m] = 1.0
        for epoch in range(
            self.max_iter_perceptron
        ):  # ← use attribute with correct name
            errors = 0
            for i in range(len(y)):
                phi_vec = phi[i]
                t = y[i]
                p = (phi_vec @ self.W_.T).argmax()
                if p != t:
                    errors += 1
                    self.W_[t] += self.lr * phi_vec
                    self.W_[p] -= self.lr * phi_vec
            if errors == 0:
                break

    def _train_ridge(self, phi: np.ndarray, y: np.ndarray) -> None:
        T = np.eye(self.n_classes, dtype=np.float32)[y]
        A = phi.T @ phi + self.lambda_ridge * np.eye(phi.shape[1], dtype=np.float32)
        B = phi.T @ T
        self.W_ = np.linalg.solve(A, B).T
