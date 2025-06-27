# PIC1: Optical and Quantum-Inspired Classification

A Python framework for developing, simulating, and testing classical, quantum-inspired, and optical machine learning models. The library provides implementations of nearest-mean and RBF classifiers, along with tools for hardware control (SLM, camera) and optical system simulation.

## Features

- **Multiple Classifier Models**:
  - `ClassicalNearestMeanClassifier`: A standard nearest-mean classifier with customizable distance metrics.
  - `QuantumNearestMeanClassifier`: A quantum-inspired version that maps inputs to density operators and uses quantum distance measures.
  - `RBFNet`: A flexible Radial Basis Function network supporting various kernel-like distance functions.
- **Advanced Distance Metrics**:
  - Classical: Euclidean distance.
  - Optical Correlators: Classical and Binary Joint-Transform Correlator (JTC), Phase Correlation.
  - Quantum: Trace distance and Fidelity-based distance for density matrices.
- **Quantum-Inspired Encodings**:
  - A suite of functions to encode classical data into quantum states (e.g., `stereographic`, `informative`, `diag_prob`).
- **Hardware Integration & Simulation**:
  - Drivers for controlling Thorlabs EXULUS SLMs and UC480-compatible cameras.
  - A LightPipes-based simulation of a binary Joint-Transform Correlator (`binary_jtc_lightpipes.py`).
  - Scripts for hardware calibration and running optical correlation experiments.

## Project Structure

- **`src/data`**: Data loaders for MNIST and Fashion-MNIST datasets.
- **`src/models`**: Core classifier implementations.
  - `ClassicalNearestMean/`: Classical Nearest-Mean Classifier.
  - `QuantumNearestMean/`: Quantum-inspired Nearest-Mean Classifier.
  - `RBF/`: Radial Basis Function (RBF) Network.
  - `utils.py`: Shared utilities, including a factory for creating distance functions.
- **`src/distance`**: Implementations of various distance and similarity metrics (JTC, quantum distances, etc.).
- **`src/encodings`**: Functions for mapping classical vectors to quantum state representations.
- **`src/hardware`**: Modules for controlling and simulating optical hardware.
  - `devices/`: Python wrappers for camera and SLM hardware drivers.
  - `simulations/`: Scripts for simulating optical systems (e.g., JTC with LightPipes).
  - `calibration/`: Utilities for calibrating hardware components like the SLM.
- **`src/tests`**: Scripts for validating models and running experiments.
  - `models/`: Test and validation scripts for the classifiers.
  - `hardware/`: Scripts for running experiments on the physical optical setup.

## Usage

### Running Model Evaluations

You can run cross-validation and testing for each model using the scripts in `src/tests/models`.

- **Classical Nearest Mean**:
  ```bash
  python3 -m src.tests.models.test_classical_nearest_mean
  ```
- **Quantum Nearest Mean**:
  ```bash
  python3 -m src.tests.models.test_quantum_nearest_mean
  ```
- **RBF Network**:
  ```bash
  python3 -m src.tests.models.test_RBF
  ```

### Running Simulations and Hardware Experiments

- **Simulated Optical Correlation**:
  ```bash
  python3 -m src.hardware.simulations.binary_jtc_lightpipes
  ```
- **Optical Correlation on Hardware**:
  ```bash
  python3 -m src.tests.hardware.correlate_digits
  ```

### Example: Training a Classifier

Here is a basic example of how to train and evaluate the `RBFNet` classifier.

```python
from src.models.RBF.rbf_network import RBFNet
from src.data.data import get_train_data, get_test_data
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load data
X_train, y_train = get_train_data()
X_test,  y_test  = get_test_data()

# Configure and train a pipeline
model = Pipeline([
    ("pca", PCA(n_components=50)),
    ("rbf", RBFNet(n_centers=100, distance_name='classical_jtc'))
])
model.fit(X_train, y_train)

# Predict and evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```