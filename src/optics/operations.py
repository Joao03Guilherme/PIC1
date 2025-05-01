import matplotlib.pyplot as plt
import numpy as np


class OpticalSystem:
    def __init__(self, name):
        self.name = name
        self.components = []

    @staticmethod
    def transpose_matrix(matrix):
        """Transpose a given matrix.
        Args:
            matrix (np.ndarray): The matrix to be transposed.
        Returns:
            np.ndarray: The transposed matrix.
        """

        return matrix.T

    def add_component(self, component):
        self.components.append(component)

    def __str__(self):
        return f"Optical System: {self.name}, Components: {', '.join(str(c) for c in self.components)}"


from scipy.fft import fft2, ifft2, fftshift


def fourier_optical_system(image):
    # Simulate 4f system: FT -> Process -> IFT
    ft_image = fftshift(fft2(image))  # Forward FT (via lens 1)

    # Optionally apply filter or axis swap (simulate spatial mask)
    processed = ft_image.T  # Transpose in frequency domain (for example)

    output = np.abs(ifft2(fftshift(processed)))  # Back to spatial (via lens 2)
    return output


processed_image = fourier_optical_system(A)

plt.imshow(processed_image, cmap="inferno")
plt.title("Output Field from Simulated 4f System")
plt.colorbar()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Original matrix A (can be non-square)
A = np.array([[1, 2], [3, 4], [5, 6]])

# Step 1: Compute SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Show SVD components
print("U:\n", U)
print("Singular values:\n", S)
print("V^T:\n", VT)

# Step 2: Invert the singular values
S_inv = np.diag(1 / S)  # Σ†
print("Σ†:\n", S_inv)

# Step 3: Construct the pseudoinverse A† = V Σ† U^T
A_pseudo = VT.T @ S_inv @ U.T
print("A† (Pseudoinverse of A):\n", A_pseudo)

# Optional: Check A * A† * A ≈ A
reconstructed_A = A @ A_pseudo @ A
print("Reconstructed A (A * A† * A):\n", reconstructed_A)

# Adaptive, optical, radial basis function neural network for handwritten digit recognition
# https://chatgpt.com/s/dr_681208a2a60c8191816c11d202dec7e2