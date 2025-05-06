import csv
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List

# -----------------------------------------------------------------------------
# Constants & file paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "mnist_csv"
TRAIN_CSV = DATA_DIR / "mnist_train.csv"
TEST_CSV = DATA_DIR / "mnist_test.csv"

MNIST_URLS = {
    TRAIN_CSV: "https://pjreddie.com/media/files/mnist_train.csv",
    TEST_CSV: "https://pjreddie.com/media/files/mnist_test.csv",
}


def load_csv(
    filepath: Union[str, Path], flatten: bool = True, return_tuple: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], List]:
    """
    Load a CSV file containing MNIST or similar data.

    Parameters:
        filepath: Path to the CSV file
        flatten: If True, returns vectors as flat arrays; otherwise, reshape to 28x28
        return_tuple: If True, returns (vectors, labels); if False, returns [(label, vector), ...]

    Returns:
        If return_tuple:
            vectors: numpy array of shape (n_samples, 784) or (n_samples, 28, 28)
            labels: numpy array of shape (n_samples,)
        Else:
            list of tuples (label, vector)
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath

    with filepath.open("r") as f:
        reader = csv.reader(f)
        data = np.asarray([list(map(int, row)) for row in reader], dtype=np.uint8)

    labels = data[:, 0].astype(np.int64)
    vectors = data[:, 1:].astype(np.float32)

    if not flatten:
        vectors = vectors.reshape(-1, 28, 28)

    if return_tuple:
        return vectors, labels
    else:
        return [
            (labels[i], vectors[i].reshape(-1, 1) if flatten else vectors[i])
            for i in range(len(labels))
        ]


def get_train_data(**kwargs):
    """Convenience function to load MNIST training data."""
    return load_csv(TRAIN_CSV, **kwargs)


def get_test_data(**kwargs):
    """Convenience function to load MNIST test data."""
    return load_csv(TEST_CSV, **kwargs)
