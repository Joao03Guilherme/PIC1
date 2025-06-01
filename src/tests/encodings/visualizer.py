import numpy as np
from ...encodings.encodings import (
    encode_diag_prob,
    encode_stereographic,
    encode_informative,
)

# Create example vectors (e.g., unit vectors, random vectors, scaled vectors)
np.random.seed(0)
X = np.random.rand(100, 5)  # 100 vectors of dimension 5

encoded_diag = np.array([encode_diag_prob(x) for x in X])
encoded_stereo = np.array([encode_stereographic(x) for x in X])
encoded_info = np.array([encode_informative(x) for x in X])

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embeddings(encoded, title):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(encoded)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(True)
    plt.show()

visualize_embeddings(encoded_diag, "Diagonal Probability Encoding (PCA)")
visualize_embeddings(encoded_stereo, "Stereographic Encoding (PCA)")
visualize_embeddings(encoded_info, "Informative Encoding (PCA)")

from sklearn.metrics import pairwise_distances

original_dists = pairwise_distances(X, metric='euclidean')
diag_dists = pairwise_distances(encoded_diag, metric='euclidean')
stereo_dists = pairwise_distances(encoded_stereo, metric='euclidean')
info_dists = pairwise_distances(encoded_info, metric='euclidean')

# Correlate original and transformed distances
def corr_distance(original, transformed, name):
    from scipy.stats import spearmanr
    corr, _ = spearmanr(original.ravel(), transformed.ravel())
    print(f"Spearman correlation for {name}: {corr:.3f}")

corr_distance(original_dists, diag_dists, "Diagonal")
corr_distance(original_dists, stereo_dists, "Stereographic")
corr_distance(original_dists, info_dists, "Informative")
