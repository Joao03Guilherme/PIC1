import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Added PCA

# Add project root to sys.path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data import get_test_data, get_train_data # Added get_train_data
from src.encodings.encodings import (
    encode_diag_prob,
    encode_informative,
    encode_stereographic,
    normalize_vector,
    compute_density_matrix_from_vector
)

# The standard encoding is given by the density matrix of the normalized vector
# The remaining encoding functions typically handle normalization of the input pixel vector internally.

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'density_matrices')

def plot_density_matrix(matrix, title, filename):
    """Helper function to plot and save a density matrix."""
    plt.figure(figsize=(8, 6))
    # Using np.real as density matrices can be complex, but imshow needs real values.
    # For density matrices, diagonal elements are real, off-diagonals can be complex.
    # Visualizing the real part is a common practice.
    img = plt.imshow(np.real(matrix), cmap='viridis', aspect='auto')
    plt.colorbar(img, label='Value')
    plt.title(title)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved density matrix figure to {filename}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load training data to fit PCA
    print("Loading training data for PCA...")
    X_train, _ = get_train_data(dataset_name='mnist')
    if X_train.shape[0] == 0:
        print("No training data loaded for PCA. Exiting.")
        return
    
    # 2. Fit PCA
    print(f"Fitting PCA to reduce to 50 components on training data of shape {X_train.shape}...")
    pca = PCA(n_components=50)
    pca.fit(X_train)
    print("PCA fitting complete.")

    # 3. Load a sample vector from the MNIST test dataset
    print("Loading test sample...")
    X_test, y_test = get_test_data(dataset_name='mnist', sample_size=1) # Get 1 sample
    if X_test.shape[0] == 0:
        print("No test data loaded. Exiting.")
        return
    
    original_sample_vector = X_test[0]
    print(f"Loaded original sample vector of shape: {original_sample_vector.shape}")

    # 4. Apply PCA to the sample vector
    # PCA expects a 2D array, so reshape if it's 1D, then select the first (and only) transformed sample.
    if original_sample_vector.ndim == 1:
        original_sample_vector_2d = original_sample_vector.reshape(1, -1)
    else:
        original_sample_vector_2d = original_sample_vector
        
    sample_vector_pca = pca.transform(original_sample_vector_2d)[0]
    print(f"Transformed sample vector (after PCA) shape: {sample_vector_pca.shape}")

    # Ensure the PCA-transformed vector is used for encodings
    sample_vector = sample_vector_pca

    encodings_to_test = [
        {
            "name": "Amplitude Encoding",
            "func": lambda vec: compute_density_matrix_from_vector(normalize_vector(vec)),
            "filename_base": "amplitude_encoding"
        },
        {
            "name": "Diagonal Probability Encoding",
            "func": lambda vec: encode_diag_prob(vec), # Handles normalization internally
            "filename_base": "diag_prob_encoding"
        },
        {
            "name": "Informative Amplitude Encoding (alpha=0.7, beta=0.3)",
            "func": lambda vec: encode_informative(vec),
            "filename_base": "informative_amplitude_encoding"
        },
        {
            "name": "Inverse Stereographic Encoding",
            "func": lambda vec: encode_stereographic(vec), # Uses default params
            "filename_base": "inv_stereo_encoding"
        }
    ]

    for encoding_info in encodings_to_test:
        print(f"Processing: {encoding_info['name']}")
        try:
            density_matrix = encoding_info["func"](sample_vector.copy()) # Use a copy
            plot_title = f"Density Matrix - {encoding_info['name']}"
            output_filename = os.path.join(OUTPUT_DIR, f"{encoding_info['filename_base']}.png")
            plot_density_matrix(density_matrix, plot_title, output_filename)
        except Exception as e:
            print(f"Error during encoding {encoding_info['name']}: {e}")

if __name__ == "__main__":
    main()