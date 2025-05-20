"""
Code to run the main application and do the classification
"""
 
# Use absolute imports instead of relative imports
from ...models.ClassicalNearestMean.c_nearestmean_network import (
    ClassicalNearestMeanClassifier,
)
from ...models.QuantumNearestMean.quantum_nearest_mean import (
    QuantumNearestMeanClassifier,
)

from ...models.RBF.rbf_network import RBFNet

from ...data.data import get_test_data, get_train_data
from ...hardware.devices.Camera import UC480Controller
from ...hardware.devices.SLM import SLMdisplay
from ...distance.OpticalJTCorrelator import OpticalJTCorrelator

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# Global variables for data caching
X_train_global = None
y_train_global = None
X_test_global = None
y_test_global = None

def load_all_data_globally():
    """Loads training and testing data into global variables if not already loaded."""
    global X_train_global, y_train_global, X_test_global, y_test_global
    if X_train_global is None or y_train_global is None:
        print("Loading training data globally...")
        X_train_global, y_train_global = get_train_data()
        print(f"Training data loaded: X_train_global shape {X_train_global.shape}, y_train_global shape {y_train_global.shape}")
    if X_test_global is None or y_test_global is None:
        print("Loading testing data globally...")
        X_test_global, y_test_global = get_test_data()
        print(f"Testing data loaded: X_test_global shape {X_test_global.shape}, y_test_global shape {y_test_global.shape}")

def test_classic_nearest_mean():
    """Test the classical nearest mean classifier with Euclidean distance"""
    global X_train_global, y_train_global, X_test_global, y_test_global
    print("\nTesting Classical Nearest Mean Classifier (Euclidean distance)...")
    # Ensure data is loaded by the main block
    if X_train_global is None or y_train_global is None or X_test_global is None or y_test_global is None:
        print("Error: Data not loaded globally. Call load_all_data_globally() in the main execution block.")
        return

    print(f"Using global X_train shape: {X_train_global.shape}, y_train shape: {y_train_global.shape}")

    model = ClassicalNearestMeanClassifier(
        distance_metric_name="euclidean", distance_squared=False, random_state=0
    )

    start_time = time.time()
    model.fit(X_train_global, y_train_global)
    fit_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test_global)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test_global, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fit time: {fit_time:.2f}s, Predict time: {predict_time:.2f}s")


def test_quantum_nearest_mean_with_optical_jtc():
    """Test the quantum nearest mean classifier with PCA and OpticalJTCorrelator"""
    global X_train_global, y_train_global, X_test_global, y_test_global
    print("\nTesting Quantum Nearest Mean Classifier with PCA and Optical JTC...")

    # Ensure data is loaded by the main block
    if X_train_global is None or y_train_global is None or X_test_global is None or y_test_global is None:
        print("Error: Data not loaded globally. Call load_all_data_globally() in the main execution block.")
        return

    print(f"Using global original data - X_train: {X_train_global.shape}, y_train: {y_train_global.shape}")

    # Apply PCA to reduce dimensions to 50 features
    n_components = 50
    print(f"Applying PCA to reduce to {n_components} components...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_global)
    X_test_pca = pca.transform(X_test_global)

    print(f"After PCA - X_train: {X_train_pca.shape}, X_test: {X_test_pca.shape}")

    # Initialize hardware resources
    print("Initializing hardware resources (SLM and camera)...")
    print("Attempting to initialize SLMdisplay...")
    slm = SLMdisplay(monitor=1, isImageLock=True)  # Adjust monitor if needed
    print("SLMdisplay initialized.")
    
    print("Attempting to initialize UC480Controller (camera)...")
    cam = UC480Controller()  # Adjust serial if needed
    print("UC480Controller (camera) initialized.")

    # Setup optical correlator with reasonable defaults
    print("Attempting to set up OpticalJTCorrelator...")
    correlator = OpticalJTCorrelator(slm=slm, cam=cam, sleep_time=0.1)
    print("OpticalJTCorrelator set up.")

    # Set camera exposure (adjust as needed for your hardware setup)
    exposure_ms = 11
    print(f"Attempting to set camera exposure to {exposure_ms}ms...")
    correlator.set_exposure(exposure_ms)  # milliseconds
    print("Camera exposure set.")

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
        y_train_subset = y_train_global[indices] # Use global y_train for subset labels

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
        y_test_subset = y_test_global[test_indices] # Use global y_test for subset labels

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

    # Load all data globally once
    load_all_data_globally()

    # Test standard classical classifier
    test_classic_nearest_mean()

    # Test quantum classifier with optical correlator
    test_quantum_nearest_mean_with_optical_jtc()
