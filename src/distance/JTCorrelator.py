from __future__ import annotations
import numpy as np
from .utils import _peak_and_shift
from typing import Tuple


EPS = 1e-8  # small value to avoid division by zero

# ──────────────────────────────────────────────────────────────────────────────
# Classical JTC  (grey-scale, un-thresholded)
# ──────────────────────────────────────────────────────────────────────────────
def classical_jtc(img1_vec, img2_vec, shape):
    img1 = np.reshape(img1_vec, shape)
    img2 = np.reshape(img2_vec, shape)

    # Concatenate left and right halves to form the joint input plane
    joint = np.hstack((img1, img2))  # shape (H, 2W)

    # Compute joint power spectrum: |F1 + F2|²
    joint_power_spectrum = np.abs(np.fft.fft2(joint)) ** 2

    # Correlation plane: inverse FFT of power spectrum
    corr = np.fft.ifft2(joint_power_spectrum)
    corr = np.fft.fftshift(np.real(corr))

    # Mask out central zero-order region and extract cross-correlation peak
    peak, (dy, dx) = _peak_and_shift(corr, shape)

    # Compute normalized similarity from the peak
    norm = np.linalg.norm(img1) * np.linalg.norm(img2)
    similarity = peak / (2 * norm) if norm else 0.0

    # Use inverse similarity as a distance
    distance = 1 / similarity

    return distance, (dy, dx), similarity, corr / (2 * norm)


def binary_jtc(img1_vec: np.ndarray,
               img2_vec: np.ndarray,
               shape: Tuple[int, int],
               spectrum_threshold: str | float = "median"):
    """
    Binary Joint-Transform Correlator (grey-scale inputs, binary spectrum)

    Parameters
    ----------
    img1_vec, img2_vec : 1-D uint8/float32
        Flattened images, same length.
    shape : (H, W)
        Original 2-D shape.
    spectrum_threshold :
        "median"  – threshold = median(|F|²)   (default)
        "mean"    – threshold = mean(|F|²)
        <float>   – threshold = float × mean(|F|²)

    Returns
    -------
    distance   : float        (≥ 1, 1 = identical)
    shift      : (dy, dx)     cross-correlation peak position
    similarity : float        (≈ 1 for identical, 0 for unrelated)
    corr       : 2-D ndarray  centred correlation plane
    """
    if img1_vec.size != img2_vec.size:
        raise ValueError("img1 and img2 must contain the same number of pixels")

    # 1) reshape
    img1 = img1_vec.astype(np.float32).reshape(shape)
    img2 = img2_vec.astype(np.float32).reshape(shape)

    # 2) joint input plane
    joint = np.hstack((img1, img2))                 # (H, 2W)

    # 3) joint power spectrum
    jps = np.abs(np.fft.fft2(joint)) ** 2

    if spectrum_threshold == "median":
        thresh = np.median(jps)
    elif spectrum_threshold == "mean":
        thresh = jps.mean()
    else:
        thresh = float(spectrum_threshold) * jps.mean()

    jps_bin = (jps >= thresh).astype(np.float32)    # 0/1 binary spectrum

    # 4) correlation plane
    corr = np.fft.ifft2(jps_bin)
    corr = np.fft.fftshift(np.real(corr))

    # 5) peak & similarity
    peak, (dy, dx) = _peak_and_shift(corr, shape)
    norm = np.linalg.norm(img1) * np.linalg.norm(img2) + EPS

    similarity = peak / (2.0 * norm)               # ∈ [0, 1]
    distance   = 1.0 / (similarity + EPS)          # ∈ [1, ∞)

    return distance, (dy, dx), similarity, corr


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
    # Recompute distances with same synthetic images
    np.random.seed(0)
    shape = (28, 28)

    # images as before
    img_base = np.random.randint(0, 255, shape).astype(np.float32)
    img_ident = img_base.copy()
    img_shift = np.roll(np.roll(img_base, 5, axis=0), 3, axis=1)
    img_noise = img_base + np.random.normal(0, 1, shape)
    img_bright = np.clip(img_base * 1.3, 0, 255)
    img_rotate = np.rot90(img_base)
    img_random = np.random.randint(0, 255, shape).astype(np.float32)

    variants = [
        img_base,
        img_ident,
        img_shift,
        img_noise,
        img_bright,
        img_rotate,
        img_random,
    ]

    eucl_dists = []
    jtc_dists = []

    for v in variants:
        eucl_dists.append(np.linalg.norm(img_base.flatten() - v.flatten()))
        jtc_dist, _, _, _ = binary_jtc(img_base.flatten(), v.flatten(), shape)
        jtc_dists.append(jtc_dist)

    # Convert to numpy arrays
    eucl_dists = np.array(eucl_dists)
    jtc_dists = np.array(jtc_dists)

    # polyfit for Euclidean = f(JTC)
    m, c = np.polyfit(jtc_dists, eucl_dists, 1)
    y_pred = m * jtc_dists + c

    # R^2
    ss_res = np.sum((eucl_dists - y_pred) ** 2)
    ss_tot = np.sum((eucl_dists - eucl_dists.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # bar plot peaks
    labels = [
        "base vs ident",
        "base vs ident",
        "base vs shift",
        "base vs noise",
        "base vs bright",
        "base vs rotate",
        "base vs random",
    ]
    # compute peaks for bar: reuse similarity computed earlier
    peaks = []
    for v in variants:
        # run jtc to get similarity
        img1 = img_base.flatten()
        img2 = v.flatten()
        _, _, sim, _ = binary_jtc(img1, img2, shape)
        peaks.append(sim)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].bar(labels[1:], peaks[1:], color="steelblue")
    ax[0].set_ylabel("Correlation Peak")
    ax[0].set_title("Correlation Peak Values")
    ax[0].tick_params(axis="x", rotation=25)

    # scatter + fit
    xfit = np.linspace(jtc_dists.min(), jtc_dists.max(), 100)
    yfit = m * xfit + c

    ax[1].scatter(jtc_dists, eucl_dists, color="blue", marker="x", label="Data points")
    ax[1].plot(xfit, yfit, "b--", label=f"Fit: y={m:.2f}x+{c:.2f}\n$R^2$={r2:.3f}")
    ax[1].set_xlabel("JTC Distance")
    ax[1].set_ylabel("Euclidean Distance")
    ax[1].set_title("Euclidean Distance vs. JTC Distance")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()
