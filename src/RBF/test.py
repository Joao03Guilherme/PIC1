import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import csv

FILENAME_TRAIN = "src/data_encoding/dataset/MNIST_CSV/mnist_train.csv"
FILENAME_TEST = "src/data_encoding/dataset/MNIST_CSV/mnist_test.csv"

def get_dataset(filename=FILENAME_TRAIN):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = np.array([list(map(int, row)) for row in reader], dtype=np.uint8)
    return [(row[1:], row[0]) for row in data]     # (vector, label)

def get_vectors(dataset):
    # ravel() flattens in case the vector already has a second dim
    return np.array([vec.ravel() for vec, _ in dataset], dtype=np.float32)

def get_labels(dataset):
    return np.array([label for _, label in dataset], dtype=np.int64)

class RBFNet:
    def __init__(self, n_centers=None, k_sigma=1.0, n_classes=10,
                 max_iter_perceptron=10, lr=0.1):
        self.n_centers   = n_centers     # None → use every sample
        self.k_sigma     = k_sigma
        self.n_classes   = n_classes
        self.max_iter_p  = max_iter_perceptron
        self.lr          = lr

    # ---------- TRAIN ----------
    def fit(self, X, y):
        X = X.astype('float32') / 255.       # (N,784) in [0,1]
        self.scaler_ = StandardScaler(with_std=False).fit(X) # just mean-centre
        X0 = self.scaler_.transform(X)

        # 1. choose centres
        self.centers_ = (MiniBatchKMeans(self.n_centers, batch_size=2048,
                                          max_iter=100, n_init='auto')
                         .fit(X0).cluster_centers_
                         if self.n_centers else X0.copy())

        # 2. widths → gamma
        d = pairwise_distances(self.centers_)
        np.fill_diagonal(d, np.inf)
        sigmas = self.k_sigma * d.min(1)
        self.gammas_ = 0.5 / np.square(sigmas + 1e-12)

        # 3. design matrix Φ  (vectorised, one shot)
        Φ = np.exp(-pairwise_distances(X0, self.centers_,
                                       squared=True) * self.gammas_)
        # 4. binary-address init
        self.W_ = np.zeros((self.n_classes, Φ.shape[1]), dtype=np.float32)
        self.W_[y, np.arange(Φ.shape[1]) % Φ.shape[1]] = 1  # works for both modes

        # 5. perceptron refinement (fully vectorised)
        for _ in range(self.max_iter_p):
            logits   = Φ @ self.W_.T
            pred     = logits.argmax(1)
            mask_err = pred != y
            if not mask_err.any(): break
            # gather Φ rows that were mis-classified once per class for speed
            for cls in range(self.n_classes):
                idx = np.where((y == cls) & mask_err)[0]
                if idx.size:
                    Δ = Φ[idx].sum(0)
                    self.W_[cls]   += self.lr * Δ
                    self.W_[pred[idx]] -= self.lr * Δ

        return self

    # ---------- PREDICT ----------
    def predict(self, X):
        X  = self.scaler_.transform(X.astype('float32') / 255.)
        Φ  = np.exp(-pairwise_distances(X, self.centers_, squared=True)
                    * self.gammas_)
        return (Φ @ self.W_.T).argmax(1)


from joblib import dump, load

dataset_train = get_dataset(FILENAME_TRAIN)
dataset_test = get_dataset(FILENAME_TEST)

X_train, y_train = get_vectors(dataset_train), get_labels(dataset_train)
X_test, y_test = get_vectors(dataset_test), get_labels(dataset_test)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

net = RBFNet(n_centers=None,    
             k_sigma=1.0,
             max_iter_perceptron=5,
             lr=0.05).fit(X_test, y_test)

dump(net, "rbf_mnist.joblib")   

pred = net.predict(X_test)
print("Accuracy =", (pred == y_test).mean())