from ...encodings.encodings import (
    normalize_vector,
    encode_informative,
    calculate_purity_from_vector,  # Keep if used elsewhere, or remove if only matrix purity is needed now
    compute_density_matrix_from_vector,
    calculate_purity_from_density_matrix,  # Ensure this is imported
)

from ...data.data import get_train_data, get_test_data
from sklearn.decomposition import PCA  # Import PCA

import numpy as np

train_vectors, train_labels = get_train_data()
test_vectors, test_labels = get_test_data()


# Apply PCA before encoding
pca = PCA(
    n_components=50, svd_solver="full", random_state=0
)  # Retain 95% of variance
X_train_pca = pca.fit_transform(train_vectors)
X_test_pca = pca.transform(test_vectors)

print(f"Original number of features: {train_vectors.shape[1]}")
print(f"Number of features after PCA: {X_train_pca.shape[1]}")

# Encode each data point
encoded_train_data = np.array([normalize_vector(x) for x in X_train_pca])
encoded_test_data = np.array([normalize_vector(x) for x in X_test_pca])

vector = encoded_test_data[0]
purity_vector = calculate_purity_from_vector(vector)
print(
    f"Purity of the vector (from encoded vector): {purity_vector:.4f}"
)  # Clarified print

# Calculate centroids (average density matrices) for each class
print("\nCalculating class centroids (average density matrices) and their purities:")
unique_classes = np.unique(train_labels)

centroids = {}
purities = {}
density_matrices = {}  # Store density matrices for inspection if needed

for class_label in unique_classes:
    # Get all samples of this class
    class_indices = train_labels == class_label
    class_samples = train_vectors[class_indices]

    if len(class_samples) == 0:
        print(
            f"  Class {class_label}: No samples found, skipping centroid calculation."
        )
        continue

    # 1. Convert each vector in the class to its density matrix
    class_density_matrices_list = []
    for sample_vector in class_samples:
        encoded_vector = normalize_vector(sample_vector)
        density_matrix = compute_density_matrix_from_vector(encoded_vector)
        class_density_matrices_list.append(density_matrix)

    # 2. Sum all density matrices and divide by the number to get the average density matrix (centroid)
    if not class_density_matrices_list:
        print(
            f"  Class {class_label}: No density matrices generated, skipping centroid calculation."
        )
        continue

    sum_of_density_matrices = np.sum(np.array(class_density_matrices_list), axis=0)
    num_density_matrices = len(class_density_matrices_list)
    average_density_matrix_centroid = sum_of_density_matrices / num_density_matrices

    # 3. Ensure the centroid is trace-normalized before purity calculation
    trace_val = np.trace(average_density_matrix_centroid)
    if np.isclose(trace_val, 0):
        print(
            f"  Class {class_label}: Centroid trace is close to zero, cannot normalize. Purity will be calculated on unnormalized matrix."
        )
        # Or handle as an error, or assign a default purity e.g., NaN or 1/dim
        normalized_centroid_for_purity = average_density_matrix_centroid
    else:
        normalized_centroid_for_purity = average_density_matrix_centroid / trace_val

    centroids[class_label] = (
        normalized_centroid_for_purity  # Store the trace-normalized centroid
    )

    # 4. Compute the purity of this average (and now trace-normalized) density matrix
    purity = calculate_purity_from_density_matrix(normalized_centroid_for_purity)
    purities[class_label] = purity

    print(f"  Class {class_label}:")
    # print(f"    Number of samples: {len(class_samples)}")
    # print(f"    Shape of average density matrix centroid: {average_density_matrix_centroid.shape}")
    # print(f"    Trace of average density matrix centroid (before normalization): {trace_val:.4f}")
    # print(f"    Trace of normalized centroid for purity: {np.trace(normalized_centroid_for_purity):.4f}")
    print(f"    Purity of average density matrix centroid: {purity:.4f}")

print("\nSummary of purities for centroids (average density matrices):")
for class_label, purity in purities.items():
    print(f"  Purity for class {class_label} centroid: {purity:.4f}")
