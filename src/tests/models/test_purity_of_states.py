from ...encodings.encodings import (
    normalize_vector,
    encode_informative,
    encode_stereographic,
    calculate_purity_from_vector,  # Keep if used elsewhere, or remove if only matrix purity is needed now
    compute_density_matrix_from_vector,
    calculate_purity_from_density_matrix,  # Ensure this is imported
)

from ...data.data import get_train_data, get_test_data
from sklearn.decomposition import PCA  # Import PCA

import numpy as np

train_vectors, train_labels = get_train_data()
test_vectors, test_labels = get_test_data()


pca = PCA(n_components=50, svd_solver="full", random_state=0)
X_train_pca = pca.fit_transform(train_vectors)
# X_test_pca = pca.transform(test_vectors) # Keep if test data is used later, otherwise can be removed for this specific task

print(f"Original number of features: {train_vectors.shape[1]}")
print(f"Number of features after PCA: {X_train_pca.shape[1]}")

encodings_to_test = [
    {
        "name": "Standard",
        "func": lambda vec: compute_density_matrix_from_vector(normalize_vector(vec)),
    },
    {
        "name": "Informative (Amplitude)",
        "func": lambda vec: compute_density_matrix_from_vector(encode_informative(vec)),
    },
    {
        "name": "Stereographic",
        "func": lambda vec: compute_density_matrix_from_vector(
            encode_stereographic(vec)
        ),
    },
]

unique_classes = np.unique(train_labels)
all_purities_summary = {}  # To store all results for a final summary

print(
    "\nCalculating class centroids (average density matrices) and their purities for different encodings:"
)

for encoding_info in encodings_to_test:
    encoding_name = encoding_info["name"]
    encoding_fn = encoding_info["func"]

    print(f"\n--- Encoding: {encoding_name} ---")

    purities_for_current_encoding = {}  # Store purities for the current encoding type

    for class_label in unique_classes:
        class_indices = train_labels == class_label
        class_samples_pca = X_train_pca[class_indices]  # Use PCA'd training data

        if len(class_samples_pca) == 0:
            print(
                f"  Class {class_label}: No samples found, skipping centroid calculation."
            )
            continue

        class_density_matrices_list = []
        for sample_vector_pca in class_samples_pca:
            # Apply the current encoding function to the PCA'd vector
            # This function is expected to return a density matrix
            density_matrix = encoding_fn(sample_vector_pca)
            class_density_matrices_list.append(density_matrix)

        if not class_density_matrices_list:
            print(
                f"  Class {class_label}: No density matrices generated, skipping centroid calculation."
            )
            continue

        sum_of_density_matrices = np.sum(np.array(class_density_matrices_list), axis=0)
        num_density_matrices = len(class_density_matrices_list)
        average_density_matrix_centroid = sum_of_density_matrices / num_density_matrices

        trace_val = np.trace(average_density_matrix_centroid)
        if (
            np.isclose(trace_val, 0) or np.isnan(trace_val) or np.isinf(trace_val)
        ):  # Added checks for nan/inf
            print(
                f"  Class {class_label}: Centroid trace is {trace_val}, cannot normalize. Purity will be calculated on unnormalized matrix (or result might be NaN)."
            )
            normalized_centroid_for_purity = average_density_matrix_centroid
        else:
            normalized_centroid_for_purity = average_density_matrix_centroid / trace_val

        # centroids_for_current_encoding[class_label] = normalized_centroid_for_purity # If you need to store centroids

        purity = calculate_purity_from_density_matrix(normalized_centroid_for_purity)
        purities_for_current_encoding[class_label] = purity

        print(f"  Class {class_label}:")
        print(f"    Purity of average density matrix centroid: {purity:.4f}")

    all_purities_summary[encoding_name] = purities_for_current_encoding

# Final summary of all purities
print("\n--- Overall Purity Summary ---")
for encoding_name_summary, class_purities_summary in all_purities_summary.items():
    print(f"Encoding: {encoding_name_summary}")
    for class_label_summary, purity_val_summary in class_purities_summary.items():
        print(
            f"  Class {class_label_summary} centroid purity: {purity_val_summary:.4f}"
        )
