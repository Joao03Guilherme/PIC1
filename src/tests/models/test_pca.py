"""
Test Principal Component Analysis (PCA) to check how many features are reduced from the original dataset.
"""

from ...data.data import get_train_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load training data
X_train, _ = get_train_data()

# Original feature count
original_features = X_train.shape[1]

# Apply PCA to retain 95% of variance
pca = PCA(n_components=0.90, svd_solver="full", random_state=0)
pca.fit(X_train)
reduced_features = pca.n_components_

# Print feature reduction summary
print(f"Original number of features: {original_features}")
print(f"Number of features after PCA (95% variance): {reduced_features}")
print(f"Features reduced: {original_features - reduced_features}")

# Compute per-pixel importance as weighted sum of squared loadings
# approximate image shape (assumes square images)
side = int(np.sqrt(original_features))
# pca.components_: shape (n_components, n_features)
# explained_variance_ratio_: length n_components
importance = np.sqrt(
    np.sum((pca.components_.T**2) * pca.explained_variance_ratio_, axis=1)
)
importance_matrix = importance.reshape(side, side)

# Plot heatmap of feature importance
plt.figure(figsize=(6, 6))
sns.heatmap(importance_matrix, cmap="viridis")
plt.title("PCA Feature Importance Heatmap (95% variance)")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()
