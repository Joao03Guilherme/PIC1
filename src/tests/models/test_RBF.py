from ...models.RBF.rbf_network import RBFNet
from ...data.data import get_test_data, get_train_data

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    make_scorer,
    confusion_matrix,
)
from pathlib import Path # Add this import

# Define ROOT as the directory containing the current test script
ROOT = Path(__file__).resolve().parent

X_train_total, y_train_total = get_train_data()
X_test_total, y_test_total = get_test_data()

# ---------------------------------------------------------------------
# Optional: subsample the huge datasets so CV runs quickly
# ---------------------------------------------------------------------
train_per_class = 150
test_per_class = 25
n_classes = 10

X_train = np.empty((n_classes * train_per_class, 784), dtype=np.float32)
y_train = np.empty((n_classes * train_per_class,), dtype=np.int64)
X_test = np.empty((n_classes * test_per_class, 784), dtype=np.float32)
y_test = np.empty((n_classes * test_per_class,), dtype=np.int64)

row_t = row_v = 0
for cls in range(n_classes):
    tr_idx = np.where(y_train_total == cls)[0][:train_per_class]
    te_idx = np.where(y_test_total == cls)[0][:test_per_class]

    next_t, next_v = row_t + train_per_class, row_v + test_per_class
    X_train[row_t:next_t] = X_train_total[tr_idx]
    y_train[row_t:next_t] = y_train_total[tr_idx]
    X_test[row_v:next_v] = X_test_total[te_idx]
    y_test[row_v:next_v] = y_test_total[te_idx]
    row_t, row_v = next_t, next_v

print("train shape:", X_train.shape, " test shape:", X_test.shape)

# ---------------------------------------------------------------------
# Build PCA → RBF pipeline
# ---------------------------------------------------------------------
# pca = PCA(n_components=0.90, svd_solver="full", random_state=0)

rbf = RBFNet(
    n_centers=50,
    k_sigma=0.5,
    distance_name="euclidean",
    distance_squared=False,
    lambda_ridge=1e-3,
    random_state=0,
)

model = Pipeline(
    [
        ("rbf", rbf),
    ]
)

# ---------------------------------------------------------------------
# k-fold cross-validation (stratified)
# ---------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

scoring = {
    "acc": "accuracy",
    "bal_acc": make_scorer(balanced_accuracy_score),
}

print("\nRunning 5-fold stratified cross-validation …")
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
