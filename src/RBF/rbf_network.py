from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple, Union

import numpy as np  # switch to CuPy manually if desired
from joblib import dump
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from JTCorrelator import (
    jtc_binary,
    jtc_classical,
    phase_corr_similarity,
)  # ↔ your C/Numba implementation

# ----------------------------------------------------------------------------
# Constants & helpers
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "mnist_csv"
TRAIN_CSV = DATA_DIR / "mnist_train.csv"
TEST_CSV = DATA_DIR / "mnist_test.csv"


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vectors, labels) where vectors are (N, 784) uint8 → float32."""
    with path.open("r") as f:
        reader = csv.reader(f)
        data = np.asarray([list(map(int, row)) for row in reader], dtype=np.uint8)
    labels = data[:, 0].astype(np.int64)
    vectors = data[:, 1:].astype(np.float32)  # 0‥255 range, no normalisation
    return vectors, labels


# ──────────────────────────────────────────────────────────────────────────────
#  Flexible distance / similarity factory
# ──────────────────────────────────────────────────────────────────────────────
def make_distance_fn(name: str = "phase", *, squared: bool = False):
    """
    Return a callable f(X, Y) compatible with sklearn.pairwise_distances.

    Parameters
    ----------
    name     : 'phase' | 'jtc_binary' | 'jtc_classical' | 'euclidean'
    squared  : if True, return f² (common for Gaussian RBF kernels)

    The function **always returns a true “distance”**:
      – larger value  →  less similar
      – non-negative
    """
    if name == "phase":

        def _d(X, Y):
            d, _, _ = phase_corr_similarity(X, Y, shape=(28, 28))
            return d

    elif name == "jtc_binary":
        # higher peak ⇒ more similar ⇒ distance should be *smaller*
        def _d(X, Y):
            peak = jtc_binary(
                X.reshape(28, 28),
                Y.reshape(28, 28),
                binarize_inputs=True,
            )
            return max(0.0, 1.0 - peak)  # invert similarity

    elif name == "jtc_classical":

        def _d(X, Y):
            peak = jtc_classical(
                X.reshape(28, 28),
                Y.reshape(28, 28),
            )
            return max(0.0, 1.0 - peak)

    elif name == "euclidean":
        # Let sklearn handle it natively
        return "euclidean" if not squared else "sqeuclidean"
    else:
        raise ValueError(f"Unknown distance name '{name}'")

    # optional squaring
    if squared:
        return lambda X, Y, *, _f=_d: _f(X, Y) ** 2
    return _d


# ----------------------------------------------------------------------------
# RBF network class
# ----------------------------------------------------------------------------
class RBFNet:
    """Gaussian-RBF or binary-JTC-kernel network."""

    def __init__(
        self,
        n_centers: Union[int, None] = None,
        k_sigma: float = 1.0,
        n_classes: int = 10,
        # output layer options
        use_perceptron: bool = False,
        max_iter_perceptron: int = 10,
        lr: float = 0.05,
        lambda_ridge: float = 1e-3,
        # kernel choice
        # misc
        min_sigma: float = 1e-2,
        distance_name: str = "phase",  # <── NEW
        distance_squared: bool = True,  # <── NEW
    ) -> None:
        # ­-- hyper-parameters
        self.n_centers = n_centers
        self.k_sigma = k_sigma
        self.n_classes = n_classes
        self.use_perceptron = use_perceptron
        self.max_iter_p = max_iter_perceptron
        self.lr = lr
        self.gammas_ = None  # (M,)
        self.l2 = lambda_ridge
        self.min_sigma = min_sigma
        self.distance_name = distance_name
        self.distance_squared = distance_squared

    # --------------------------- training -----------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFNet":
        """Train the network and return *self*."""
        # Make a copy; keep raw 0‥255 values for JTC distance.
        X_raw = X.astype(np.float32, copy=False)

        # 1. choose centres -------------------------------------------------
        if self.n_centers is None:
            print("Using all unique samples as centres …")
            self.centers_ = np.unique(X_raw, axis=0)
        else:
            print(f"Clustering into {self.n_centers} centres …")
            self.centers_ = (
                MiniBatchKMeans(
                    self.n_centers,
                    batch_size=4096,
                    max_iter=100,
                    n_init="auto",
                    random_state=0,
                )
                .fit(X_raw)
                .cluster_centers_.astype(np.float32)
            )

        dist_fn = make_distance_fn(self.distance_name, squared=self.distance_squared)

        # 2. compute distances ---------------------------------------------
        d2 = pairwise_distances(self.centers_, metric=dist_fn, n_jobs=-1)
        np.fill_diagonal(d2, np.inf)
        nn2 = d2.min(axis=1)  # shape (M,)
        sigma2 = np.maximum((self.k_sigma**2) * nn2, self.min_sigma**2)
        self.gammas_ = 0.5 / sigma2

        # 3. Build design matrix ------------------------------------------
        phi = np.exp(
            -pairwise_distances(X_raw, self.centers_, metric=dist_fn, n_jobs=-1)
            * self.gammas_[None, :]
        )
        # 4. train output layer --------------------------------------------
        if self.use_perceptron:
            self._train_perceptron(phi, y)
        else:
            self._train_ridge(phi, y)
        return self

    # --------------------------- predictors ----------------------------------
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

    # -------------------  output-layer training methods ----------------------
    def _train_perceptron(self, phi: np.ndarray, y: np.ndarray) -> None:
        print("Training output layer with perceptron …")
        m = phi.shape[1]
        self.W_ = np.zeros((self.n_classes, m), dtype=np.float32)
        self.W_[y, np.arange(len(y)) % m] = 1.0

        for epoch in range(self.max_iter_p):
            errors = 0
            for i in range(len(y)):
                phi_vec = phi[i]
                t = y[i]
                p = (phi_vec @ self.W_.T).argmax()
                if p != t:
                    errors += 1
                    self.W_[t] += self.lr * phi_vec
                    self.W_[p] -= self.lr * phi_vec
            print(f"  epoch {epoch}: errors = {errors}")
            if errors == 0:
                break

    def _train_ridge(self, phi: np.ndarray, y: np.ndarray) -> None:
        print("Training output layer with ridge-regularised least squares …")
        T = np.eye(self.n_classes, dtype=np.float32)[y]  # one-hot
        A = phi.T @ phi + self.l2 * np.eye(phi.shape[1], dtype=np.float32)
        B = phi.T @ T
        self.W_ = np.linalg.solve(A, B).T  # shape (k, m)


# ----------------------------------------------------------------------------
# Script mode
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    if not TEST_CSV.exists() or not TRAIN_CSV.exists():
        raise FileNotFoundError(
            "MNIST CSV files not found. Place csv files in 'mnist_csv/'."
        )

    X_train_total, y_train_total = _load_csv(TRAIN_CSV)
    X_test_total, y_test_total = _load_csv(TEST_CSV)

    # -------------------------------------------------------------------------
    # how many samples of each class?
    train_per_class = 150  # 10×60 = 600 total
    test_per_class = 25  # 10×10 = 100 total
    n_classes = 10

    # allocate the right size up-front
    X_train = np.zeros((n_classes * train_per_class, 784), dtype=np.float32)
    y_train = np.zeros((n_classes * train_per_class,), dtype=np.int64)
    X_test = np.zeros((n_classes * test_per_class, 784), dtype=np.float32)
    y_test = np.zeros((n_classes * test_per_class,), dtype=np.int64)

    row_t, row_v = 0, 0
    for cls in range(n_classes):
        # indices of all samples belonging to this class
        tr_idx = np.where(y_train_total == cls)[0][:train_per_class]
        te_idx = np.where(y_test_total == cls)[0][:test_per_class]

        # copy them into the next free slice
        next_t = row_t + train_per_class
        next_v = row_v + test_per_class
        X_train[row_t:next_t] = X_train_total[tr_idx]
        y_train[row_t:next_t] = y_train_total[tr_idx]
        X_test[row_v:next_v] = X_test_total[te_idx]
        y_test[row_v:next_v] = y_test_total[te_idx]
        row_t, row_v = next_t, next_v  # advance cursors

    print("train shape:", X_train.shape, " test shape:", X_test.shape)

    net = RBFNet(
        n_centers=500,
        use_perceptron=False,
        lambda_ridge=1e-3,
        distance_squared=False,
        k_sigma=0.5,
        distance_name="phase",  # 'jtc_binary' | 'jtc_classical' | 'euclidean' | 'phase'
    ).fit(X_train, y_train)

    dump(net, ROOT / "rbf_mnist.joblib")
    print("Model saved to rbf_mnist.joblib")

    preds = net.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"Test accuracy: {acc * 100:.2f} %")

    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for cls in range(n_classes):
        # Get indices for this class
        cls_indices = np.where(y_test == cls)[0]
        cls_correct = (preds[cls_indices] == y_test[cls_indices]).sum()
        cls_total = len(cls_indices)
        print(
            f"Class {cls}: {cls_correct} out of {cls_total} correct ({cls_correct/cls_total*100:.2f}%)"
        )
