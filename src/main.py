"""
Code to run the main application and do the classification
"""

from models.ClassicalNearestMean.c_nearestmean_network import (
    ClassicalNearestMeanClassifier,
)
from src.models.QuantumNearestMean.quantum_nearestmean_network import (
    QuantumNearestMeanClassifier,
)

from models.RBF.rbf_network import RBFNet

from data.data import get_test_data, get_train_data
from hardware.devices.Camera import UC480Controller
from hardware.devices.SLM import SLMdisplay
from distance.OpticalJTCorrelator import OpticalJTCorrelator

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import time


def test_classic_nearest_mean():
    """Test the classical nearest mean classifier with Euclidean distance"""
    print("\nTesting Classical Nearest Mean Classifier (Euclidean distance)...")
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    model = ClassicalNearestMeanClassifier(
        distance_metric_name="euclidean", distance_squared=False, random_state=0
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")


def test_quantum_nearest_mean_with_optical_jtc():
    """Test the quantum nearest mean classifier with PCA and OpticalJTCorrelator"""
    print("\nTesting Quantum Nearest Mean Classifier with PCA and Optical JTC...")

    # Load data
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()

    print(f"Original data - X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Apply PCA to reduce dimensions to 50 features
    n_components = 50
    print(f"Applying PCA to reduce to {n_components} components...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"After PCA - X_train: {X_train_pca.shape}, X_test: {X_test_pca.shape}")

    # Initialize hardware resources
    print("Initializing hardware resources (SLM and camera)...")
    slm = SLMdisplay(monitor=1, isImageLock=True)  # Adjust monitor if needed
    cam = UC480Controller()  # Adjust serial if needed

    # Setup optical correlator with reasonable defaults
    print("Setting up OpticalJTCorrelator...")
    correlator = OpticalJTCorrelator(slm=slm, cam=cam, sleep_time=0.1)

    # Set camera exposure (adjust as needed for your hardware setup)
    correlator.set_exposure(11)  # milliseconds

    # Try to set an appropriate ROI for the camera (adjust based on your setup)
    try:
        # Get sensor size to determine appropriate ROI
        width, height = cam.detector_size
        correlator.set_roi(
            x=width // 4,  # Start from quarter of the way in
            y=height // 4,  # Start from quarter of the way down
            width=width // 2,  # Use half the width
            height=height // 2,  # Use half the height
        )
    except Exception as e:
        print(f"Warning: Could not set ROI: {e}")

    try:
        # Setup Quantum Nearest Mean Classifier with optical correlator
        print("Creating and training the Quantum Nearest Mean Classifier...")
        model = QuantumNearestMeanClassifier(
            encoding="stereographic",  # Maps features to density matrix
            distance="optical_classical_jtc",  # Use the optical correlator
            optical_correlator=correlator,
        )

        # Train on a small subset first as a test
        # Using a small subset because optical processing can be slow
        subset_size = 100
        indices = np.random.choice(len(X_train_pca), subset_size, replace=False)
        X_train_subset = X_train_pca[indices]
        y_train_subset = y_train[indices]

        print(f"Training on a subset of {subset_size} examples...")
        start_time = time.time()
        model.fit(X_train_subset, y_train_subset)
        fit_time = time.time() - start_time
        print(f"Fit time: {fit_time:.2f}s")

        # Test on a small subset as well
        test_subset_size = 20
        test_indices = np.random.choice(
            len(X_test_pca), test_subset_size, replace=False
        )
        X_test_subset = X_test_pca[test_indices]
        y_test_subset = y_test[test_indices]

        print(f"Testing on a subset of {test_subset_size} examples...")
        start_time = time.time()
        y_pred_subset = model.predict(X_test_subset)
        predict_time = time.time() - start_time

        accuracy = accuracy_score(y_test_subset, y_pred_subset)
        print(f"Subset accuracy: {accuracy:.4f}")
        print(f"Predict time for {test_subset_size} samples: {predict_time:.2f}s")

        # Confusion matrix
        cm = confusion_matrix(y_test_subset, y_pred_subset)
        print("Confusion matrix:")
        print(cm)

    except Exception as e:
        print(f"Error during quantum optical test: {e}")
    finally:
        # Always clean up hardware resources
        print("Closing hardware resources...")
        try:
            correlator.close()
        except Exception as e:
            print(f"Error closing correlator: {e}")


if __name__ == "__main__":
    print("Testing classifiers...")

    # Test standard classical classifier
    test_classic_nearest_mean()

    # Test quantum classifier with optical correlator
    test_quantum_nearest_mean_with_optical_jtc()
