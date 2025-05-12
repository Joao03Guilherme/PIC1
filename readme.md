# PIC1 Classification Library

A Python library implementing a variety of nearest-mean classifiers and related utilities.

## Subpackages

- **src/models/RBF**  
  Implements `RBFNet` for Gaussian or correlation‐based radial basis function networks (ridge or perceptron output).

- **src/models/ClassicalNearestMean**  
  Implements `ClassicalNearestMeanClassifier` that computes class medoids using a specified distance metric (e.g., classical JTC).

- **src/models/QuantumNearestMean**  
  Implements `QuantumNearestMeanClassifier`, a quantum‐inspired nearest mean classifier mapping inputs to density operators.

- **src/models/utils**  
  Utility functions, including `make_distance_fn` to build custom distance metrics (phase correlation, JTC, Euclidean, etc.).

- **src/distance**  
  Contains `JTCorrelator.py` with classical and binary JTC implementations, and phase correlation metrics.

- **src/data**  
  Data loaders (`get_train_data`, `get_test_data`) for your datasets.

- **src/tests/models**  
  Test scripts for each classifier:
  - `test_RBF.py`  
    Run: `python3 -m src.tests.models.test_RBF`
  - `test_classical_nearest_mean.py`  
    Run: `python3 -m src.tests.models.test_classical_nearest_mean`
  - `test_quantum_nearest_mean.py`  
    Run: `python3 -m src.tests.models.test_quantum_nearest_mean`

## Installation

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd PIC1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Example: train and evaluate the RBF network on a small sample
```python
from src.models.RBF.rbf_network import RBFNet
from src.data.data import get_train_data, get_test_data

X_train, y_train = get_train_data()
X_test,  y_test  = get_test_data()

# configure and train
model = RBFNet(n_centers=100, distance_name='classical_jtc')
model.fit(X_train, y_train)

# predict and evaluate
y_pred = model.predict(X_test)
```