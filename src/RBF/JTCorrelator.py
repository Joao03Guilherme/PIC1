from __future__ import annotations
import numpy as np
from typing import Tuple


import numpy as np
from typing import Tuple


def classical_jtc(img1_vec, img2_vec, shape):
    """
    Classical Joint Transform Correlator (JTC) for two grayscale images.
    - img1_vec, img2_vec: Flattened 1D numpy arrays of the two images (must be same length).
    - shape: Tuple (H, W) giving the height and width to reshape the images.
    Returns: distance, (dy, dx), peak_corr
      distance: Mean squared error after aligning img2 to img1 (lower = more similar).
      (dy, dx): Estimated shift of img2 relative to img1 (in pixels).
      peak_corr: Peak cross-correlation value (a similarity score, 1.0 = perfect match).
    """
    # Reshape vectors to 2D images
    img1 = np.reshape(img1_vec, shape)
    img2 = np.reshape(img2_vec, shape)
    
    # Combine the two images side by side into a single image
    combined_img = np.concatenate((img1, img2), axis=1)

    # Compute 2D FFT of the combined image
    F_combined = np.fft.fft2(combined_img)

    # Calculate the intensity of the fourier transform
    intensity = np.abs(F_combined) ** 2

    # Perform the inverse FFT to get the correlation result
    corr = np.fft.ifft2(intensity)

    return np.abs(corr)

def binarize(img, treshold="median"):
    """
    Calculates the median pixel value of the image and binarizes it.
    - img: 2D numpy array representing the image.
    - treshold: Method to calculate the treshold. Currently only "median" is supported.
    Returns: 2D numpy array with binary values (-1 or 1).
    """

    if treshold == "median":
        treshold_value = np.median(img)

    else:
        if isinstance(treshold, (int, float)):
            treshold_value = treshold
        else:
            raise ValueError("Invalid treshold value. Use 'median' or a numeric value.")
    
    # Binarize the image: -1 for pixels below median, 1 for pixels above
    binary_img = np.where(img < treshold_value, -1, 1)

    return binary_img

def binary_jtc(img1_vec, img2_vec, shape):
    """
    Binary Joint Transform Correlator (JTC) for two binary images.
    - img1_vec, img2_vec: Flattened 1D numpy arrays of the two images (must be same length).
    - shape: Tuple (H, W) giving the height and width to reshape the images.
    Returns: distance, (dy, dx), peak_corr
      distance: Mean squared error after aligning img2 to img1 (lower = more similar).
      (dy, dx): Estimated shift of img2 relative to img1 (in pixels).
      peak_corr: Peak cross-correlation value (a similarity score, 1.0 = perfect match).
    """
    # Reshape vectors to 2D images
    img1 = np.reshape(img1_vec, shape)
    img2 = np.reshape(img2_vec, shape)

    combined_img = np.concatenate((img1, img2), axis=1)
    combined_reference = np.concatenate((img1, img1), axis=1)

    # Calculate the treshold (media of the pixels of img1 + img1)
    median_value = float(np.median(combined_reference))
    
    # Combine the two images side by side into a single image
    combined_img = binarize(np.concatenate((img1, img2), axis=1), treshold=median_value)

    # Compute 2D FFT of the combined image
    F_combined = np.fft.fft2(combined_img)

    # Calculate the intensity of the fourier transform
    intensity = binarize(np.abs(F_combined) ** 2)

    # Perform the inverse FFT to get the correlation result
    corr = np.fft.ifft2(intensity)

    return np.abs(corr)



def phase_corr_similarity(img1_vec, img2_vec, shape):
    """
    Estimate a shift-invariant distance between two grayscale images.
    - img1_vec, img2_vec: Flattened 1D numpy arrays of the two images (must be same length).
    - shape: Tuple (H, W) giving the height and width to reshape the images.
    Returns: distance, (dy, dx), peak_corr
      distance: Mean squared error after aligning img2 to img1 (lower = more similar).
      (dy, dx): Estimated shift of img2 relative to img1 (in pixels).
      peak_corr: Peak cross-correlation value (a similarity score, 1.0 = perfect match).
    """
    # Reshape vectors to 2D images
    img1 = np.reshape(img1_vec, shape)
    img2 = np.reshape(img2_vec, shape)
    # Compute 2D FFTs
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    # Compute normalized cross-power spectrum (avoid division by zero)
    eps = 1e-8
    cross_power = F1 * np.conj(F2)
    cross_power /= np.abs(cross_power) + eps
    # Inverse FFT to get cross-correlation
    corr = np.fft.ifft2(cross_power)
    # Find peak correlation and location
    corr_mag = np.abs(corr)
    peak_corr = corr_mag.max()
    peak_idx = np.unravel_index(np.argmax(corr_mag), corr_mag.shape)
    # Calculate shifts (account for wrap-around)
    dy, dx = peak_idx
    H, W = shape
    if dy > H // 2:  # wrap-around adjustment
        dy -= H
    if dx > W // 2:
        dx -= W
    # Align img2 by the estimated shift (using roll for circular shift)
    aligned_img2 = np.roll(np.roll(img2, -dy, axis=0), -dx, axis=1)
    # Compute MSE as distance
    diff = img1 - aligned_img2
    distance = np.mean(diff**2)
    return distance, (dy, dx), peak_corr, corr


# ---------------------------------------------------------------------
# Quick demo (executed only when run as a script)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    np.random.seed(0)

    shape = (28, 28)

    # -----------------------------------------------------------------
    # Generate a variety of related / unrelated images
    # -----------------------------------------------------------------
    img_base   = np.random.randint(0, 255, shape).astype(np.float32)

    img_ident  = img_base.copy()                                   # identical
    img_shift  = np.roll(np.roll(img_base, 5, axis=0), 3, axis=1)  # shifted
    img_noise  = img_base + np.random.normal(0, 1, shape)          # noisy
    img_bright = np.clip(img_base * 1.3, 0, 255)                   # brighter
    img_rotate = np.rot90(img_base)                                # 90° rotation
    img_random = np.random.randint(0, 255, shape).astype(np.float32)  
    

    correlation_binary = binary_jtc(
        img_base.flatten(), img_ident.flatten(), shape
    )
    correlation_classical = classical_jtc(
        img_base.flatten(), img_ident.flatten(), shape
    )
    _,_,c,correlation_phase = phase_corr_similarity(
        img_base.flatten(), img_ident.flatten(), shape
    )


    fig = plt.figure(figsize=(18, 6))

    # Get shapes for plotting
    Hb, Wb = correlation_binary.shape
    Hp, Wp = correlation_phase.shape

    # Meshgrids
    Xb, Yb = np.meshgrid(np.arange(Wb), np.arange(Hb))  # for binary and classical
    Xp, Yp = np.meshgrid(np.arange(Wp), np.arange(Hp))  # for phase

    # Plot 1: Binary JTC Correlation
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(Xb, Yb, correlation_binary, cmap='plasma', edgecolor='none')
    ax1.set_title('Binary JTC Correlation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Amplitude')

    # Plot 2: Classical JTC Correlation
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(Xb, Yb, correlation_classical, cmap='viridis', edgecolor='none')
    ax2.set_title('Classical JTC Correlation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Amplitude')

    # Plot 3: Phase Correlation
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(Xp, Yp, np.abs(correlation_phase), cmap='inferno', edgecolor='none')
    ax3.set_title('Phase-Only Correlation')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Amplitude')

    plt.tight_layout()
    plt.show()



    """

    images = {
        "ident" : img_ident,
        "shift" : img_shift,
        "bright": img_bright,
        "rotate": img_rotate,
        "random": img_random,
    }

    # -----------------------------------------------------------------
    # Compare each variant to the base image
    # -----------------------------------------------------------------
    labels, est_distances, real_distances, corr_peaks = [], [], [], []

    for name, img in images.items():
        label = f"base vs {name}"
        dist_est, (dy, dx), pc = phase_corr_similarity(
            img_base.flatten(), img.flatten(), shape
        )
        real_dist = np.linalg.norm(img_base.flatten() - img.flatten())

        labels.append(label)
        est_distances.append(dist_est)
        real_distances.append(real_dist)
        corr_peaks.append(pc)

        print(f"{label}:")
        print(f"  Estimated distance: {dist_est:.2f}")
        print(f"  Real distance:      {real_dist:.2f}")
        print(f"  Correlation peak:   {pc:.2f}")
        print(f"  Shift detected:     ({dy}, {dx})\n")

    # -----------------------------------------------------------------
    # Linear fit for scatter plot
    # -----------------------------------------------------------------
    m, b = np.polyfit(real_distances, est_distances, 1)
    print(f"Linear fit: estimated ≈ {m:.3f} × real + {b:.3f}")

    # Points for fit line (two endpoints are enough)
    x_fit = np.array([min(real_distances), max(real_distances)])
    y_fit = m * x_fit + b

    # -----------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig)          # 1 row × 2 columns

    # ── Plot 1: Correlation-peak values ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(labels, corr_peaks)
    ax1.set_ylabel('Correlation Peak')
    ax1.set_title('Correlation Peak Values')
    ax1.set_ylim(0, 1)
    ax1.set_xticklabels(labels, rotation=20, ha='right')

    # ── Plot 2: Scatter – estimated vs real distance + fit line ──────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(real_distances, est_distances, label='Data points')
    ax2.plot(x_fit, y_fit, linestyle='--', label=f'Fit: y={m:.2f}x+{b:.2f}')
    ax2.set_xlabel('Real Distance')
    ax2.set_ylabel('Estimated Distance')
    ax2.set_title('Estimated vs Real Distance')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    """
