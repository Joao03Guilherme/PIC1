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

import numpy as np

X_train, y_train = get_train_data()
X_test, y_test = get_test_data()

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

model = ClassicalNearestMeanClassifier(
    distance_metric_name="euclidean", distance_squared=False, random_state=0
)