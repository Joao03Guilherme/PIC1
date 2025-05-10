import csv
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List

# Define DATA_DIR relative to the current file's location
DATA_DIR = Path(__file__).resolve().parent / "mnist_csv"
TRAIN_CSV = DATA_DIR / "mnist_train.csv"
TEST_CSV = DATA_DIR / "mnist_test.csv"


def load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vectors, labels) where vectors are (N, 784) uint8 → float32."""
    with path.open("r") as f:
        reader = csv.reader(f)
        data = np.asarray([list(map(int, row)) for row in reader], dtype=np.uint8)
    labels = data[:, 0].astype(np.int64)
    vectors = data[:, 1:].astype(np.float32)  # 0‥255 range, no normalisation
    return vectors, labels


def get_train_data(**kwargs):
    """Convenience function to load MNIST training data."""
    return load_csv(TRAIN_CSV)


def get_test_data(**kwargs):
    """Convenience function to load MNIST test data."""
    return load_csv(TEST_CSV)
