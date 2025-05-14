from ...models.RBF.rbf_network import RBFNet
from ...data.data import get_test_data, get_train_data

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)  # Added train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    make_scorer,
    confusion_matrix,
)
from sklearn.decomposition import PCA  # Add PCA import
from pathlib import Path  # Add this import

# Define ROOT as the directory containing the current test script
ROOT = Path(__file__).resolve().parent

X_train_total, y_train_total = get_train_data()
X_test_total, y_test_total = get_test_data()

# Define the percentage of the dataset to use (e.g., 0.1 for 10%)
sample_percentage = 0.025

# Create smaller, stratified training subset
_, X_train, _, y_train = train_test_split(
    X_train_total,
    y_train_total,
    test_size=sample_percentage,
    stratify=y_train_total,
    random_state=0,
)

# Create smaller, stratified testing subset
_, X_test, _, y_test = train_test_split(
    X_test_total,
    y_test_total,
    test_size=sample_percentage,
    stratify=y_test_total,
    random_state=0,
)

# ---------------------------------------------------------------------
# Build PCA → RBF pipeline
# ---------------------------------------------------------------------
# Print the shapes of the training and testing sets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Create PCA component
pca = PCA(n_components=0.95, svd_solver="full", random_state=0)

# Create RBF component
rbf = RBFNet(
    n_centers=100,
    k_sigma=0.5,
    distance_name="classical_jtc",
    distance_squared=False,
    lambda_ridge=1e-3,
    # We let shape be determined automatically from the PCA-transformed data
    # This will create a shape as square as possible
    shape=None,
    random_state=0,
)

model = Pipeline(
    [
        ("pca", pca),  # PCA preprocessing step
        ("rbf", rbf),
    ]
)

# Let's see how many features PCA would retain with our dataset
pca_check = PCA(n_components=0.95, svd_solver="full", random_state=0)
X_train_pca = pca_check.fit_transform(X_train)
print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of features after PCA: {X_train_pca.shape[1]} (retaining 95% variance)")

# ---------------------------------------------------------------------
# k-fold cross-validation (stratified)
# ---------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

scoring = {
    "acc": "accuracy",
    "bal_acc": make_scorer(balanced_accuracy_score),
}

print("\nRunning 10-fold stratified cross-validation …")
cv_res = cross_validate(
    model,
    X_train,
    y_train,
    scoring=scoring,
    cv=cv,
    n_jobs=-1,
    return_train_score=False,
)

print(" per-fold accuracy     :", np.round(cv_res["test_acc"], 4))
print(" per-fold bal. accuracy:", np.round(cv_res["test_bal_acc"], 4))
print(
    " mean ± std accuracy   : %.4f ± %.4f"
    % (cv_res["test_acc"].mean(), cv_res["test_acc"].std())
)
print(
    " mean ± std bal. acc.  : %.4f ± %.4f"
    % (cv_res["test_bal_acc"].mean(), cv_res["test_bal_acc"].std())
)

# Evaluate the model on the test set
print("\nEvaluating on the test set …")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Save confusion matrix as an image
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
conf_matrix_image_path = ROOT / "confusion_matrix.png"
plt.savefig(conf_matrix_image_path)
plt.close()
print(f"Confusion matrix image saved to {conf_matrix_image_path}")
