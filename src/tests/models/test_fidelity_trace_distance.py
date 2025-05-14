import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split  # Import train_test_split

from ...data.data import get_train_data, get_test_data
from ...encodings.encodings import (
    encode_stereographic,
    compute_density_matrix_from_vector,
)
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
        value_inside_sqrt = 0  # Clamp to zero due to potential floating point issues
    return np.sqrt(value_inside_sqrt)

# Add these functions after your existing functions but before main():

def generate_random_pure_state_dm(dim=2):
    """Generate a random pure state density matrix of dimension dim x dim."""
    # Generate a random complex vector of dimension dim
    vec = np.random.normal(size=(dim, 2)).view(complex).flatten()
    # Normalize the vector
    vec = vec / np.sqrt(np.sum(np.abs(vec)**2))
    # Create density matrix |ψ⟩⟨ψ|
    rho = np.outer(vec, vec.conj())
    return rho

def generate_random_mixed_state_dm(dim=2, mixing_param=None):
    """
    Generate a random mixed state as a convex combination of random pure states.
    
    Args:
        dim: Dimension of the Hilbert space
        mixing_param: If provided, uses this as the mixing parameter.
                      If None, generates a random mixing parameter.
    """
    # Generate two random pure states
    pure_state1 = generate_random_pure_state_dm(dim)
    pure_state2 = generate_random_pure_state_dm(dim)
    
    # Ensure the second state is different from the first
    retry_count = 0
    while np.allclose(pure_state1, pure_state2) and retry_count < 10:
        pure_state2 = generate_random_pure_state_dm(dim)
        retry_count += 1
    
    # Generate a random mixing parameter if not provided
    if mixing_param is None:
        # Use a random parameter that ensures the state is truly mixed
        # (avoid values too close to 0 or 1)
        mixing_param = 0.3 + 0.4 * np.random.random()  # Between 0.3 and 0.7
    
    # Create the mixed state as a convex combination
    mixed_state = mixing_param * pure_state1 + (1 - mixing_param) * pure_state2
    
    return mixed_state

def plot_random_states_distance_comparison():
    """
    Generate density matrices using stereographic encoding, create pure and mixed centroids,
    and compare how the fidelity distance and trace distance behave.
    """
    print("\n" + "="*80)
    print("Analyzing real-valued density matrices with pure and mixed centroids")
    print("="*80)
    
    # Parameters
    feature_dim = 150  # Dimension of random feature vectors before encoding
    num_pure_centroids = 5  # Number of pure state centroids to generate
    num_mixed_centroids = 5  # Number of mixed state centroids to generate
    samples_per_centroid = 10  # Number of samples per centroid
    
    # Function to generate a random feature vector
    def generate_random_features(dim):
        return np.random.normal(size=dim)
    
    # First, generate pure and mixed centroids using stereographic encoding
    centroids = []  # Each item is (centroid_dm, centroid_type, original_vector)
    
    print(f"Generating {num_pure_centroids} pure centroids and {num_mixed_centroids} mixed centroids...")
    
    # Generate pure centroids
    for i in range(num_pure_centroids):
        # For pure centroids, just encode a random feature vector
        feature_vec = generate_random_features(feature_dim)
        encoded_vec = encode_stereographic(feature_vec)
        centroid_dm = compute_density_matrix_from_vector(encoded_vec)
        centroids.append((centroid_dm, 'pure', feature_vec))
    
    # Generate mixed centroids
    for i in range(num_mixed_centroids):
        # For mixed centroids, average two encoded vectors
        feature_vec1 = generate_random_features(feature_dim)
        feature_vec2 = generate_random_features(feature_dim)
        
        encoded_vec1 = encode_stereographic(feature_vec1)
        encoded_vec2 = encode_stereographic(feature_vec2)
        
        dm1 = compute_density_matrix_from_vector(encoded_vec1)
        dm2 = compute_density_matrix_from_vector(encoded_vec2)
        
        # Mix with a random parameter between 0.3 and 0.7 to ensure it's truly mixed
        mix_param = 0.3 + 0.4 * np.random.random()
        mixed_dm = mix_param * dm1 + (1 - mix_param) * dm2
        
        # For reference we store the average feature vector as the "original"
        avg_feature = mix_param * feature_vec1 + (1 - mix_param) * feature_vec2
        centroids.append((mixed_dm, 'mixed', avg_feature))
    
    # Generate samples around each centroid
    all_samples = []  # Each item is (sample_dm, centroid_idx, centroid_type)
    
    print(f"Generating {samples_per_centroid} samples around each centroid...")
    
    for idx, (_, centroid_type, centroid_feature) in enumerate(centroids):
        for _ in range(samples_per_centroid):
            # Add noise to the original feature vector
            noise_level = 0.1 + 0.1 * np.random.random()  # Between 0.1 and 0.2
            noisy_feature = centroid_feature + noise_level * generate_random_features(feature_dim)
            
            # Encode the noisy feature vector
            encoded_noisy = encode_stereographic(noisy_feature)
            noisy_dm = compute_density_matrix_from_vector(encoded_noisy)
            
            all_samples.append((noisy_dm, idx, centroid_type))
    
    # Calculate distances between samples and centroids
    print("Calculating distances between samples and centroids...")
    
    fidelity_distances = []
    trace_distances = []
    relationship_types = []  # 'Sample-Pure' or 'Sample-Mixed'
    
    for sample_dm, centroid_idx, centroid_type in all_samples:
        centroid_dm = centroids[centroid_idx][0]
        
        # Calculate distances using the same functions as in main()
        td = calculate_trace_distance_dms(sample_dm, centroid_dm)
        fd = calculate_fidelity_distance_dms(sample_dm, centroid_dm)
        
        fidelity_distances.append(fd)
        trace_distances.append(td)
        relationship_types.append(f'Sample-{centroid_type.capitalize()}')
    
    # Convert to numpy arrays
    fidelity_distances = np.array(fidelity_distances)
    trace_distances = np.array(trace_distances)
    
    # Create scatter plot
    plt.figure(figsize=(12, 9))
    
    # Define colors for each relationship type
    color_map = {
        'Sample-Pure': 'blue',
        'Sample-Mixed': 'red'
    }
    
    # Plot points colored by relationship type
    for rel_type in np.unique(relationship_types):
        mask = np.array(relationship_types) == rel_type
        plt.scatter(
            fidelity_distances[mask],
            trace_distances[mask],
            c=color_map[rel_type],
            label=rel_type,
            cmap="viridis",
            alpha=0.7,
            s=50
        )
    
    # Linear regression for all points
    x = fidelity_distances
    y = trace_distances
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color="black", linewidth=2)
    
    plt.text(
        0.05,
        0.95,
        f"y = {m:.3f}x + {b:.3f}\n$R^2$ = {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        color="black",
    )
    
    plt.xlabel("Fidelity Distance (sample to centroid)", fontsize=12)
    plt.ylabel("Trace Distance (sample to centroid)", fontsize=12)
    plt.title(
        "Comparison of Distance Metrics for Samples to Pure vs. Mixed Centroids",
        fontsize=14,
    )
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics for each relationship type
    print("\nStatistics by relationship type:")
    for rel_type in np.unique(relationship_types):
        mask = np.array(relationship_types) == rel_type
        fd_subset = fidelity_distances[mask]
        td_subset = trace_distances[mask]
        
        # Calculate correlation
        if len(fd_subset) > 1:
            corr = np.corrcoef(fd_subset, td_subset)[0, 1]
            print(f"{rel_type}: {len(fd_subset)} samples, correlation = {corr:.4f}")
            print(f"  Mean Fidelity Distance: {fd_subset.mean():.4f}")
            print(f"  Mean Trace Distance: {td_subset.mean():.4f}")

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
    if len(np.unique(y_train_orig)) > 1:  # Stratify only if more than 1 class
        X_train, _, y_train, _ = train_test_split(
            X_train_orig,
            y_train_orig,
            train_size=subset_fraction,
            stratify=y_train_orig,
            random_state=42,
        )
    else:  # If only one class, stratification is not needed/possible, just take a fraction
        X_train, _, y_train, _ = train_test_split(
            X_train_orig, y_train_orig, train_size=subset_fraction, random_state=42
        )

    # Stratified subset for testing data
    if len(np.unique(y_test_orig)) > 1:
        X_test, _, y_test, _ = train_test_split(
            X_test_orig,
            y_test_orig,
            train_size=subset_fraction,
            stratify=y_test_orig,
            random_state=42,
        )
    else:
        X_test, _, y_test, _ = train_test_split(
            X_test_orig, y_test_orig, train_size=subset_fraction, random_state=42
        )

    print(
        f"Original training samples: {len(X_train_orig)}, Subset training samples: {len(X_train)}"
    )
    print(
        f"Original test samples: {len(X_test_orig)}, Subset test samples: {len(X_test)}"
    )

    # 2. PCA (applied to the subset)
    print("Applying PCA...")
    pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
    X_train_pca = pca.fit_transform(X_train)  # Fit on the training subset
    X_test_pca = pca.transform(X_test)  # Transform the test subset
    print(
        f"Original features (from subset): {X_train.shape[1]}, PCA features: {X_train_pca.shape[1]}"
    )

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

        fd = calculate_fidelity_distance_dms(
            test_density_matrix, true_class_centroid_dm
        )
        fidelity_distances_to_true_centroid.append(fd)

        test_sample_true_labels_for_plot.append(true_label)

    if not trace_distances_to_true_centroid or not fidelity_distances_to_true_centroid:
        print("No distances were calculated. Cannot plot.")
        return

    # 5. Plot Results
    print("Plotting results...")
    plt.figure(figsize=(12, 9))  # Increased figure size
    scatter = plt.scatter(
        fidelity_distances_to_true_centroid,
        trace_distances_to_true_centroid,
        c=test_sample_true_labels_for_plot,
        cmap="viridis",
        alpha=0.7,
        s=50,
    )

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
    plt.plot(x_line, y_line, color="red", linewidth=2)
    plt.text(
        0.05,
        0.95,
        f"y = {m:.3f}x + {b:.3f}\n$R^2$ = {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        color="red",
    )

    plt.xlabel("Fidelity Distance (to true class centroid)", fontsize=12)
    plt.ylabel("Trace Distance (to true class centroid)", fontsize=12)
    plt.title(
        "Comparison of Fidelity Distance and Trace Distance for Test Samples",
        fontsize=14,
    )

    handles, legend_labels_from_scatter = scatter.legend_elements(
        prop="colors", alpha=0.7
    )

    # Ensure correct labels for the legend
    # Get unique labels present in the actual plotted data
    unique_plotted_labels = np.unique(test_sample_true_labels_for_plot)
    final_legend_labels = [f"Class {l}" for l in unique_plotted_labels]

    if len(handles) == len(final_legend_labels):
        plt.legend(
            handles,
            final_legend_labels,
            title="True Class",
            fontsize=10,
            title_fontsize=12,
        )
    else:
        # Fallback if legend elements don't match unique labels (should not happen if c is correct)
        plt.colorbar(scatter, label="True Class Label")

    plt.grid(True, linestyle="--", alpha=0.5)  # Adjusted grid style
    plt.tight_layout()
    plt.show()
    print(f"Plotted {len(fidelity_distances_to_true_centroid)} points.")
    
    # Add this new call:
    plot_random_states_distance_comparison()


if __name__ == "__main__":
    main()
