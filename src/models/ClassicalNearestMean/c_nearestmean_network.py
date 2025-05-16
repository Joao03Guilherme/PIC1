import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances
from ..utils import make_distance_fn


class ClassicalNearestMeanClassifier(BaseEstimator, ClassifierMixin):
    """
    Classical Nearest Mean Classifier using a specified distance metric.
    Defaults to using classical Joint Transform Correlator (JTC) distance.
    """

    def __init__(
        self,
        distance_metric_name: str = "classical_jtc",
        distance_squared: bool = False,
        random_state: int = 0,
    ):
        """
        Initialize the classifier.
        Args:
            distance_metric_name (str): Name of the distance metric to use.
                                        Supported by make_distance_fn from rbf_network.
                                        Defaults to "classical_jtc".
            distance_squared (bool): Whether to use the squared version of the distance.
                                     Defaults to False. The make_distance_fn for classical_jtc
                                     returns the correlation peak height, which is a similarity.
                                     To use it as a distance, often 1 - similarity or -similarity is used.
                                     However, make_distance_fn for 'classical_jtc' returns 'd' which is
                                     already treated as a distance (e.g. smaller is better).
                                     The 'squared' parameter's effect depends on make_distance_fn's implementation.
        """
        self.distance_metric_name = distance_metric_name
        self.distance_squared = distance_squared
        self.random_state = random_state
        self.dist_fn_ = None
        self.centroids_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier by computing class centroids (means).
        The input vectors are reshaped to be as close to a square as possible
        internally by the distance function. No padding is applied.
        """
        self.classes_ = np.unique(y)
        Xp = X.astype(np.float32, copy=False)

        n_features = Xp.shape[1]

        # Determine H, W such that H * W = n_features and H, W are as close as possible
        h_candidate = int(np.sqrt(n_features))
        while n_features % h_candidate != 0:
            h_candidate -= 1
            if (
                h_candidate == 0
            ):  # Should not happen if n_features > 0, but as a safeguard
                h_candidate = 1  # Fallback for prime or small n_features
                break
        H = h_candidate
        W = n_features // H
        self.image_shape_ = (H, W)

        # Compute centroids (1D vectors of length n_features)
        # No padding is applied to centroids.
        self.centroids_ = np.array(
            [Xp[y == c_label].mean(axis=0) for c_label in self.classes_],
            dtype=np.float32,
        )

        # Create distance function that expects 1D inputs of length n_features
        # and will reshape them internally to self.image_shape_
        self.dist_fn_ = make_distance_fn(
            name=self.distance_metric_name,
            squared=self.distance_squared,
            shape=self.image_shape_,  # (H,W)
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Args:
            X (np.ndarray): Data to predict, shape (n_samples, n_features).
        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,).
        """
        if (
            not hasattr(self, "centroids_")
            or not hasattr(self, "classes_")
            or self.centroids_ is None  # Explicitly check for None
            or self.centroids_.shape[0] == 0
        ):
            raise RuntimeError(
                "The classifier has not been fitted yet or no centroids were learned."
            )

        X_processed = X.astype(np.float32, copy=False)

        distances = pairwise_distances(
            X_processed, self.centroids_, metric=self.dist_fn_, n_jobs=-1
        )
        return self.classes_[np.argmin(distances, axis=1)]
