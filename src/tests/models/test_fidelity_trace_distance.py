import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split # Import train_test_split

from ...data.data import get_train_data, get_test_data
from ...encodings.encodings import encode_stereographic, compute_density_matrix_from_vector
from ...distance.JTCorrelator import classical_jtc

def calculate_trace_distance_dms(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Computes 0.5 * ||dm1 - dm2||_1 (trace norm)."""
    diff = dm1 - dm2
    s = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(s)

def calculate_fidelity_distance_dms(dm1: np.ndarray, dm2: np.ndarray) -> float:
    """Computes sqrt(1 - F^2) where F is JTC similarity."""
    if dm1.shape != dm2.shape:
        raise ValueError("Density matrices must have the same shape for JTC.")
    
    similarity_F = classical_jtc(dm1.flatten(), dm2.flatten(), shape=dm1.shape)[2]
    value_inside_sqrt = 1 - similarity_F**2
    if value_inside_sqrt < 0:
        value_inside_sqrt = 0 # Clamp to zero due to potential floating point issues
    return np.sqrt(value_inside_sqrt)

def main():
    # 1. Load Data
    X_train_orig, y_train_orig = get_train_data()
    X_test_orig, y_test_orig = get_test_data()

    # Create stratified subsets (e.g., 20% of original data)
    # Use a smaller percentage if the dataset is very large and processing is still slow
    subset_fraction = 0.035
    print(f"Using a {subset_fraction*100:.0f}% stratified subset of the data.")

    # Stratified subset for training data
    # We only need the subset, so the other part can be discarded (assigned to _)
    if len(np.unique(y_train_orig)) > 1: # Stratify only if more than 1 class
        X_train, _, y_train, _ = train_test_split(
            X_train_orig, y_train_orig, 
            train_size=subset_fraction, 
            stratify=y_train_orig, 
            random_state=42
        )
    else: # If only one class, stratification is not needed/possible, just take a fraction
        X_train, _, y_train, _ = train_test_split(
            X_train_orig, y_train_orig, 
            train_size=subset_fraction, 
            random_state=42
        )

    # Stratified subset for testing data
    if len(np.unique(y_test_orig)) > 1:
        X_test, _, y_test, _ = train_test_split(
            X_test_orig, y_test_orig, 
            train_size=subset_fraction, 
            stratify=y_test_orig, 
            random_state=42
        )
    else:
        X_test, _, y_test, _ = train_test_split(
            X_test_orig, y_test_orig, 
            train_size=subset_fraction, 
            random_state=42
        )
    
    print(f"Original training samples: {len(X_train_orig)}, Subset training samples: {len(X_train)}")
    print(f"Original test samples: {len(X_test_orig)}, Subset test samples: {len(X_test)}")

    # 2. PCA (applied to the subset)
    print("Applying PCA...")
    pca = PCA(n_components=0.95, svd_solver='full', random_state=0)
    X_train_pca = pca.fit_transform(X_train) # Fit on the training subset
    X_test_pca = pca.transform(X_test)     # Transform the test subset
    print(f"Original features (from subset): {X_train.shape[1]}, PCA features: {X_train_pca.shape[1]}")

    # 3. Calculate Quantum Centroids from training data subset
    print("Calculating quantum centroids...")
    unique_labels = np.unique(y_train)
    centroids_dm = {} 

    for label in unique_labels:
        class_samples_pca = X_train_pca[y_train == label]
        if len(class_samples_pca) == 0:
            continue
        
        class_density_matrices = []
        for sample_pca in class_samples_pca:
            encoded_vector = encode_stereographic(sample_pca)
            density_matrix = compute_density_matrix_from_vector(encoded_vector)
            class_density_matrices.append(density_matrix)
        
        if not class_density_matrices:
            continue

        centroid_dm_for_class = np.mean(np.array(class_density_matrices), axis=0)
        centroids_dm[label] = centroid_dm_for_class
    print(f"Calculated {len(centroids_dm)} centroids.")

    # 4. Calculate Distances for Test Samples to their True Class Centroid
    print("Calculating distances for test samples...")
    trace_distances_to_true_centroid = []
    fidelity_distances_to_true_centroid = []
    test_sample_true_labels_for_plot = []

    for i, test_sample_pca in enumerate(X_test_pca):
        true_label = y_test[i]

        if true_label not in centroids_dm:
            continue

        encoded_test_vector = encode_stereographic(test_sample_pca)
        test_density_matrix = compute_density_matrix_from_vector(encoded_test_vector)
        true_class_centroid_dm = centroids_dm[true_label]

        td = calculate_trace_distance_dms(test_density_matrix, true_class_centroid_dm)
        trace_distances_to_true_centroid.append(td)

        fd = calculate_fidelity_distance_dms(test_density_matrix, true_class_centroid_dm)
        fidelity_distances_to_true_centroid.append(fd)
        
        test_sample_true_labels_for_plot.append(true_label)

    if not trace_distances_to_true_centroid or not fidelity_distances_to_true_centroid:
        print("No distances were calculated. Cannot plot.")
        return

    # 5. Plot Results
    print("Plotting results...")
    plt.figure(figsize=(12, 9)) # Increased figure size
    scatter = plt.scatter(fidelity_distances_to_true_centroid, 
                          trace_distances_to_true_centroid, 
                          c=test_sample_true_labels_for_plot, 
                          cmap='viridis', alpha=0.7, s=50)

    # Linear regression line and R² (numpy polyfit)
    x = np.array(fidelity_distances_to_true_centroid)
    y = np.array(trace_distances_to_true_centroid)
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red', linewidth=2)
    plt.text(
        0.05,
        0.95,
        f"y = {m:.3f}x + {b:.3f}\n$R^2$ = {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        color='red'
    )

    plt.xlabel("Fidelity Distance (to true class centroid)", fontsize=12)
    plt.ylabel("Trace Distance (to true class centroid)", fontsize=12)
    plt.title("Comparison of Fidelity Distance and Trace Distance for Test Samples", fontsize=14)
    
    handles, legend_labels_from_scatter = scatter.legend_elements(prop="colors", alpha=0.7)
    
    # Ensure correct labels for the legend
    # Get unique labels present in the actual plotted data
    unique_plotted_labels = np.unique(test_sample_true_labels_for_plot)
    final_legend_labels = [f"Class {l}" for l in unique_plotted_labels]

    if len(handles) == len(final_legend_labels):
         plt.legend(handles, final_legend_labels, title="True Class", fontsize=10, title_fontsize=12)
    else:
        # Fallback if legend elements don't match unique labels (should not happen if c is correct)
        plt.colorbar(scatter, label='True Class Label')

    plt.grid(True, linestyle='--', alpha=0.5) # Adjusted grid style
    plt.tight_layout()
    plt.show()
    print(f"Plotted {len(fidelity_distances_to_true_centroid)} points.")

if __name__ == "__main__":
    main()
