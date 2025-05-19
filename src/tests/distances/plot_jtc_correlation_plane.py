import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

# Add project root to sys.path to allow importing from src
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.distance.JTCorrelator import classical_jtc

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "jtc_correlation_planes")
IMAGE_SHAPE = (32, 32)  # Define a base shape for the images, e.g., 32x32


def create_and_save_3d_correlation_plot(correlation_plane, filename):
    H, W_double = correlation_plane.shape  # Correlation plane is typically (H, 2W)

    # Create a meshgrid for the x and y coordinates
    x_coords = np.arange(W_double)
    y_coords = np.arange(H)
    X, Y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    ax.plot_surface(X, Y, correlation_plane, cmap="viridis", edgecolor="none")

    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_zlabel("Correlation (Normalized)")

    # Optional: Adjust viewing angle for better visualization of peaks
    ax.view_init(elev=30, azim=45)

    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved 3D correlation plot to {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Generate Sample Images ---
    # Original Base Image (e.g., a simple pattern or random noise)
    img_base = np.random.rand(*IMAGE_SHAPE) * 200 + 50  # Ensure some intensity
    img_base = img_base.astype(np.float32)

    # Shifted version of the original image
    shift_y, shift_x = (
        IMAGE_SHAPE[0] // 4,
        IMAGE_SHAPE[1] // 4,
    )  # Shift by a fraction of image size
    img_shifted = np.roll(np.roll(img_base, shift_y, axis=0), shift_x, axis=1)

    # Completely random image (uncorrelated with the base image)
    img_random = np.random.rand(*IMAGE_SHAPE) * 255
    img_random = img_random.astype(np.float32)

    # Flatten images for the classical_jtc function
    img_base_vec = img_base.flatten()
    img_shifted_vec = img_shifted.flatten()
    img_random_vec = img_random.flatten()

    # --- Plot 1: Original Image vs. Shifted Version ---
    print("Processing: Original vs. Shifted Image")
    try:
        # The last returned value is the normalized correlation plane
        _, _, _, corr_plane_shifted = classical_jtc(
            img_base_vec, img_shifted_vec, IMAGE_SHAPE
        )
        plot_filename_shifted = os.path.join(
            OUTPUT_DIR, "original_vs_shifted_corr_plane.png"
        )
        create_and_save_3d_correlation_plot(corr_plane_shifted, plot_filename_shifted)
    except Exception as e:
        print(f"Error processing 'Original vs. Shifted': {e}")

    # --- Plot 2: Original Image vs. Random Image ---
    print("\nProcessing: Original vs. Random Image")
    try:
        _, _, _, corr_plane_random = classical_jtc(
            img_base_vec, img_random_vec, IMAGE_SHAPE
        )
        plot_filename_random = os.path.join(
            OUTPUT_DIR, "original_vs_random_corr_plane.png"
        )
        create_and_save_3d_correlation_plot(corr_plane_random, plot_filename_random)
    except Exception as e:
        print(f"Error processing 'Original vs. Random': {e}")


if __name__ == "__main__":
    main()
