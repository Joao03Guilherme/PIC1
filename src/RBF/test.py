import csv
import numpy as np
import os

# Number of classes/centers (change this to control the model)
N_CLASSES = 100

def get_dataset():
    with open("src/data_encoding/dataset/MNIST_CSV/mnist_train.csv", "r") as file:
        reader = csv.reader(file)
        data = np.array([list(map(int, row)) for row in reader])

    dataset = []
    for row in data:
        label = row[0]
        vector = np.array(row[1:]).reshape(-1, 1)
        dataset.append((label, vector))
    return dataset


class RbfClassifier:
    def __init__(self, dataset=[], n_centers=10, dimensions=None):
        self.weights = []
        self.centers = []
        self.gammas = []
        self.ncenters = n_centers
        self.dataset = dataset
        self.dimensions = dimensions
        self.normalized_dataset = []
        self.normalization_factor = 255

    def normalize_dataset(self, force=False):
        """
        Normalize the dataset to the range [0, 1]
        """

        if self.dataset is None:
            raise ValueError("Dataset not provided")

        if force:
            self.normalized_dataset = []

        if len(self.normalized_dataset) != 0:
            return

        self.dimensions = self.dataset[0][1].shape[0]
        for i in range(len(self.dataset)):
            self.normalized_dataset.append(
                (self.dataset[i][0], self.dataset[i][1] / self.normalization_factor)
            )

    def distance_squared(self, x1, x2):
        """
        Calculate the difference between two column vectors (np.array)
        """

        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape")
        return np.sum((x1 - x2) ** 2)

    def update_dataset(self, dataset):
        """
        Update the dataset
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None")

        self.dataset = dataset
        self.normalize_dataset(force=True)

    def _calculate_clusters(self, dataset, centers):
        """
        Calculate the clusters based on the dataset and centers
        """

    def k_means_clustering(self, n_clusters):
        """
        Implement Lloyd's algorithm to find the cluster centers
        """

        if self.dataset is None:
            raise ValueError("Dataset not provided")

        if n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")

        if n_clusters > len(self.dataset):
            raise ValueError(
                "Number of clusters cannot exceed the number of data points"
            )

        if len(self.normalized_dataset) == 0:
            self.normalize_dataset()

        # Randomly select n_clusters points as initial cluster centers
        initial_indices = np.random.choice(
            len(self.normalized_dataset), n_clusters, replace=False
        )
        centers = np.array(
            [self.normalized_dataset[i][1].flatten() for i in initial_indices]
        )

        iteration = 0
        while True:
            print(
                f"Iteration {iteration}: Computing distances and assigning clusters..."
            )

            # Assign each data point to the nearest cluster center (vectorized)
            data_points = np.array(
                [point.flatten() for _, point in self.normalized_dataset]
            )
            distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2)
            cluster_indices = np.argmin(distances, axis=1)

            # Group points by cluster
            clusters = {i: [] for i in range(n_clusters)}
            for idx, cluster_index in enumerate(cluster_indices):
                clusters[cluster_index].append(self.normalized_dataset[idx])

            # Update cluster centers (vectorized)
            new_centers = []
            for i in range(n_clusters):
                if len(clusters[i]) == 0:
                    # If a cluster is empty, reinitialize its center randomly
                    new_centers.append(data_points[np.random.randint(len(data_points))])
                else:
                    cluster_points = np.array(
                        [point.flatten() for _, point in clusters[i]]
                    )
                    new_centers.append(np.mean(cluster_points, axis=0))

            new_centers = np.array(new_centers)

            # Print the difference between new_centers and centers
            diff = np.linalg.norm(new_centers - centers)
            print(f"Difference between new_centers and centers: {diff}")

            # Check for convergence (if centers do not change)
            if np.allclose(new_centers, centers):
                print(f"Convergence reached after {iteration} iterations.")
                break

            centers = new_centers
            iteration += 1

        self.centers = centers
        return clusters

    def rbf(self, x, center, gamma):
        return np.exp(-gamma * np.sum((x - center) ** 2))

    def get_weights(self):
        if len(self.centers) == 0:
            raise ValueError("Centers must be computed before getting weights.")
        if len(self.normalized_dataset) == 0:
            self.normalize_dataset()

        print("Building RBF design matrix...")
        X = np.array([point.flatten() for _, point in self.normalized_dataset])
        y = np.array([label for label, _ in self.normalized_dataset])
        n_samples = X.shape[0]
        n_centers = len(self.centers)
        if (
            not hasattr(self, "gammas")
            or self.gammas is None
            or len(self.gammas) != n_centers
        ):
            self.gammas = np.ones(n_centers)
        Phi = np.zeros((n_samples, n_centers))
        for i in range(n_samples):
            for j in range(n_centers):
                Phi[i, j] = self.rbf(X[i], self.centers[j], self.gammas[j])
        print("Computing pseudoinverse and solving least squares...")
        Phi_pinv = np.linalg.pinv(Phi)
        self.weights = Phi_pinv @ y
        print("Weights computed.")
        return self.weights

    def choose_gammas(self, max_iter=20, lr=0.1):
        if len(self.centers) == 0:
            raise ValueError("Centers must be computed before choosing gammas.")
        if len(self.normalized_dataset) == 0:
            self.normalize_dataset()

        X = np.array([point.flatten() for _, point in self.normalized_dataset])
        y = np.array([label for label, _ in self.normalized_dataset])
        n_samples = X.shape[0]
        n_centers = len(self.centers)
        if (
            not hasattr(self, "gammas")
            or self.gammas is None
            or len(self.gammas) != n_centers
        ):
            self.gammas = np.ones(n_centers)

        for iteration in range(max_iter):
            # Build design matrix
            Phi = np.zeros((n_samples, n_centers))
            for i in range(n_samples):
                for j in range(n_centers):
                    Phi[i, j] = self.rbf(X[i], self.centers[j], self.gammas[j])
            # Compute weights for current gamma
            Phi_pinv = np.linalg.pinv(Phi)
            weights = Phi_pinv @ y
            # Compute predictions
            y_pred = Phi @ weights
            # Compute error
            error = y_pred - y
            mse = np.mean(error**2)
            print(f"Iteration {iteration}: MSE = {mse}")
            # Compute gradient w.r.t. each gamma
            grad_gamma = np.zeros(n_centers)
            for j in range(n_centers):
                for i in range(n_samples):
                    diff = X[i] - self.centers[j]
                    rbf_val = Phi[i, j]
                    grad_gamma[j] += (
                        2 * error[i] * weights[j] * rbf_val * (-np.sum(diff**2))
                    )
            grad_gamma /= n_samples
            # Update gamma
            self.gammas -= lr * grad_gamma
            self.gammas = np.maximum(self.gammas, 1e-6)  # Ensure gamma stays positive
        print("Optimized gammas:", self.gammas)
        return self.gammas

    def save_parameters(self, filepath, W=None):
        np.savez(
            filepath, weights=self.weights, gammas=self.gammas, centers=self.centers, W=W
        )
        print(f"Model parameters saved to {filepath}")

    def load_parameters(self, filepath):
        data = np.load(filepath)
        self.weights = data["weights"]
        self.gammas = data["gammas"]
        self.centers = data["centers"]
        W = data["W"] if "W" in data else None
        print(f"Model parameters loaded from {filepath}")
        return W


class RbfNetworkClassifier:
    def __init__(self, centers, gammas, weights, W):
        self.centers = centers
        self.gammas = gammas
        self.weights = weights
        self.W = W  # shape: (10, n_centers)

    def rbf_layer(self, x):
        return np.array(
            [
                np.exp(-self.gammas[j] * np.sum((x - self.centers[j]) ** 2))
                for j in range(len(self.centers))
            ]
        )

    def predict(self, x):
        # RBF layer
        phi = self.rbf_layer(x)
        # Linear combination with weights
        y = np.dot(self.weights, phi)
        # Winner-takes-all layer
        output = np.dot(self.W, phi)
        predicted_label = np.argmax(output)
        return predicted_label, output


def perceptron_refine_W(W, centers, gammas, X, y_true, n_classes=10, lr=0.01, epochs=10):
    n_centers = centers.shape[0]
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            phi = np.array([
                np.exp(-gammas[j] * np.sum((X[i] - centers[j]) ** 2))
                for j in range(n_centers)
            ])
            output = np.dot(W, phi)
            pred = np.argmax(output)
            true = y_true[i]
            if pred != true:
                errors += 1
                # Update only the true and predicted class rows
                W[true] += lr * phi
                W[pred] -= lr * phi
        print(f"Perceptron epoch {epoch}: errors = {errors}")
        if errors == 0:
            break
    return W


def train_and_save(classifier, param_file, n_classes):
    clusters = classifier.k_means_clustering(n_clusters=n_classes)
    classifier.choose_gammas(max_iter=20, lr=0.01)
    classifier.get_weights()
    n_centers = classifier.centers.shape[0]
    W = np.zeros((n_classes, n_centers))
    for i in range(n_centers):
        labels = [label for label, _ in clusters[i]]
        if labels:
            digit = max(set(labels), key=labels.count)
            W[digit, i] = 1
    X = np.array([point.flatten() for _, point in classifier.normalized_dataset])
    y_true = np.array([label for label, _ in classifier.normalized_dataset])
    W = perceptron_refine_W(W, classifier.centers, classifier.gammas, X, y_true, n_classes=n_classes, lr=0.1, epochs=5)
    classifier.save_parameters(param_file, W=W)
    return W


def evaluate(classifier, W):
    X = np.array([point.flatten() for _, point in classifier.normalized_dataset])
    y_true = np.array([label for label, _ in classifier.normalized_dataset])
    rbf_net = RbfNetworkClassifier(classifier.centers, classifier.gammas, classifier.weights, W)
    correct = 0
    for i in range(len(X)):
        pred, _ = rbf_net.predict(X[i])
        if pred == y_true[i]:
            correct += 1
    accuracy = correct / len(X)
    print(f"Classification accuracy: {accuracy * 100:.2f}%")
    print("Final gammas:", classifier.gammas)
    print("Final weights:", classifier.weights)
    print("Final W matrix:", W)


dataset = get_dataset()
dimensions = dataset[0][1].shape[0]
classifier = RbfClassifier(dataset=dataset, n_centers=N_CLASSES, dimensions=dimensions)
classifier.normalize_dataset(force=True)
param_file = "src/RBF/rbf_params.npz"

if os.path.exists(param_file):
    W = classifier.load_parameters(param_file)
    if W is None:
        # For backward compatibility if W is not in the file
        W = train_and_save(classifier, param_file, N_CLASSES)
else:
    W = train_and_save(classifier, param_file, N_CLASSES)

evaluate(classifier, W)
