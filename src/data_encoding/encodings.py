import csv
import numpy as np
import matplotlib.pyplot as plt


def get_dataset():
    with open("src/data_encoding/dataset/MNIST_CSV/mnist_test.csv", "r") as file:
        reader = csv.reader(file)
        data = np.array([list(map(int, row)) for row in reader])

    dataset = {}
    for row in data:
        label = row[0]
        vector = np.array(row[1:]).reshape(-1, 1)
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(vector)
    return dataset


def vector_to_image(vector, filename="image.png"):
    """
    Receive a vector f and generate a 2d image and save it to a file
    """

    vector = vector.reshape(28, 28)  # Reshape to 28x28 for MNIST
    vector = np.flipud(vector)  # Flip the image vertically
    plt.imsave(filename, vector, cmap="gray")  # Save as grayscale image


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


def is_density_matrix(rho):
    if not np.allclose(rho, np.conj(rho.T)):
        print("Matrix is not Hermitian")
        return False

    eigenvalues = np.linalg.eigvals(rho)
    if np.any(eigenvalues < -1e-10):  # allow small numerical errors
        print("Matrix is not positive semi-definite")
        return False

    if not np.isclose(np.trace(rho), 1):
        print("Trace is not 1")
        return False

    return True


def gram_matrix_encoding(data_vector):
    data_vector = normalize_vector(data_vector)
    gram_matrix = np.dot(data_vector, data_vector.T)
    rho = gram_matrix / np.trace(gram_matrix)
    return rho

if __name__ == "__main__":
    dataset = get_dataset()
    print("Sample vector (label 1):")
    print(dataset[1][1])

    data_vector = dataset[1][1]
    rho = gram_matrix_encoding(data_vector)

    print("\nDensity matrix:")
    print(rho)

    print("\nIs valid density matrix?")
    print(is_density_matrix(rho))

