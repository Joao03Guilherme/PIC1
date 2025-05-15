import csv
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List, Literal

# Define paths for both datasets
FASHION_DATA_DIR = Path(__file__).resolve().parent / "fashion_mnist_csv"
FASHION_TRAIN_CSV = FASHION_DATA_DIR / "fashion-mnist_train.csv"
FASHION_TEST_CSV = FASHION_DATA_DIR / "fashion-mnist_test.csv"

MNIST_DATA_DIR = Path(__file__).resolve().parent / "mnist_csv"
MNIST_TRAIN_CSV = MNIST_DATA_DIR / "mnist_train.csv"
MNIST_TEST_CSV = MNIST_DATA_DIR / "mnist_test.csv"

# Dataset name mapping
DATASET_PATHS = {
    "fashion": (FASHION_TRAIN_CSV, FASHION_TEST_CSV),
    "mnist": (MNIST_TRAIN_CSV, MNIST_TEST_CSV),
}

# Fashion MNIST class names for reference
FASHION_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Standard MNIST class names for reference
MNIST_CLASS_NAMES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vectors, labels) where vectors are (N, 784) uint8 → float32."""
    with path.open("r") as f:
        reader = csv.reader(f)
        data = np.asarray([list(map(int, row)) for row in reader], dtype=np.uint8)
    labels = data[:, 0].astype(np.int64)
    vectors = data[:, 1:].astype(np.float32)  # 0‥255 range, no normalisation
    return vectors, labels


def get_dataset(dataset_name: str = "mnist") -> Tuple[str, list]:
    """Get the name and class names for a specific dataset.

    Args:
        dataset_name: Either "fashion" or "mnist"

    Returns:
        Tuple of dataset name and list of class names
    """
    if dataset_name.lower() == "fashion":
        return "Fashion MNIST", FASHION_CLASS_NAMES
    elif dataset_name.lower() == "mnist":
        return "Standard MNIST", MNIST_CLASS_NAMES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'fashion' or 'mnist'.")


def get_train_data(dataset: str = "mnist", **kwargs):
    """Load training data from the specified dataset.

    Args:
        dataset: Which dataset to load ("fashion" or "mnist")
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        Tuple of (feature vectors, labels)
    """
    if dataset.lower() not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'fashion' or 'mnist'.")

    train_path = DATASET_PATHS[dataset.lower()][0]  # Get training path for dataset
    return load_csv(train_path)


def get_test_data(dataset: str = "mnist", **kwargs):
    """Load test data from the specified dataset.

    Args:
        dataset: Which dataset to load ("fashion" or "mnist")
        **kwargs: Additional arguments (reserved for future use)

    Returns:
        Tuple of (feature vectors, labels)
    """
    if dataset.lower() not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'fashion' or 'mnist'.")

    test_path = DATASET_PATHS[dataset.lower()][1]  # Get test path for dataset
    return load_csv(test_path)
