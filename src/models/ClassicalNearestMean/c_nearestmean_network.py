import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import pairwise_distances

# Assuming c_nearestmean_network.py is in src/models/ClassicalNearestMean/
# and rbf_network.py is in src/models/RBF/
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier by computing class centroids (means) using simple averaging.
        """
        self.classes_ = np.unique(y)
        Xp = X.astype(np.float32, copy=False)

        # Euclidean-based centroids (simple arithmetic mean)
        centroids = [Xp[y == c].mean(axis=0) for c in self.classes_]
        self.centroids_ = np.array(centroids, dtype=np.float32)

        # Initialize custom distance for prediction
        self.dist_fn_ = make_distance_fn(name=self.distance_metric_name, squared=self.distance_squared)
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
            or self.centroids_.shape[0] == 0
        ):
            raise RuntimeError(
                "The classifier has not been fitted yet or no centroids were learned (e.g., empty training data)."
            )

        X_processed = X.astype(np.float32, copy=False)

        distances = pairwise_distances(
            X_processed, self.centroids_, metric=self.dist_fn_, n_jobs=-1
        )
        return self.classes_[np.argmin(distances, axis=1)]
