from __future__ import annotations
import numpy as np
from typing import Tuple


import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────
# 1)  fast global-maximum locator (no masking, robust for identical img)
# ─────────────────────────────────────────────────────────────────────
def _peak_abs(corr_plane: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """
    Return (peak_val, (row, col)) of the *absolute* maximum in corr_plane.
    """
    idx = np.argmax(np.abs(corr_plane))
    peak_val = float(np.abs(corr_plane.flat[idx]))
    H, W = corr_plane.shape
    return peak_val, (idx // W, idx % W)


# ─────────────────────────────────────────────────────────────────────
# 2)  Vander-Lugt (phase-only) correlator with *Euclidean* distance
#     — identical signature to phase_corr_similarity
# ─────────────────────────────────────────────────────────────────────
def vlc_similarity(
    img1_vec: np.ndarray,
    img2_vec: np.ndarray,
    shape: Tuple[int, int],
    *,
    eps: float = 1e-8,
) -> Tuple[float, Tuple[int, int], float]:
    """
    Software model of a single-SLM Vander-Lugt phase-only correlator.

    Returns
    -------
    distance   : Euclidean norm  ‖img1 − aligned(img2)‖₂   (same scale as your
                 “real distance” print-out).
    (dy, dx)   : integer shift (rows, cols) that best aligns img2 to img1
    peak_corr  : correlation-peak height normalised to [0,1]
    """
    H, W = shape
    img1 = img1_vec.reshape(shape).astype(np.float32)
    img2 = img2_vec.reshape(shape).astype(np.float32)

    # 1. Fourier transforms
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)

    # 2. Phase-only matched filter  H = exp(-j·arg F1)
    H_pof = np.exp(-1j * np.angle(F1 + eps))

    # 3. Correlation plane (real because we use a conjugate filter)
    corr = np.fft.ifft2(F2 * np.conj(H_pof)).real
    corr = np.fft.fftshift(corr)  # centre the DC term

    # 4. strongest peak (centre is *not* masked ⇒ works for identical imgs)
    peak_val, (py, px) = _peak_abs(corr)

    # 5. convert peak position to wrap-around shift
    dy = py - H // 2
    dx = px - W // 2
    if dy > H // 2:
        dy -= H
    if dy < -H // 2:
        dy += H
    if dx > W // 2:
        dx -= W
    if dx < -W // 2:
        dx += W

    # 6. align img2 and compute **Euclidean** distance
    img2_aligned = np.roll(np.roll(img2, -dy, axis=0), -dx, axis=1)
    distance = float(np.linalg.norm(img1 - img2_aligned))  # ‖·‖₂

    # 7. normalised peak height   (perfect match → 1.0)
    peak_corr = float(peak_val / (np.sum(np.abs(corr)) + eps))

    return distance, (dy, dx), peak_corr


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
    return distance, (dy, dx), peak_corr


# ---------------------------------------------------------------------
# Quick demo (executed only when run as a script)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    shape = (28, 28)

    # Generate three random images with each pixel going from 0 to 255
    img1 = np.random.randint(0, 255, shape)
    img2 = np.random.randint(0, 255, shape)
    img3 = np.random.randint(0, 255, shape)

    # Flatten the images to 1-D arrays
    vec1 = img1.flatten()
    vec2 = img2.flatten()
    vec3 = img3.flatten()

    # Compute distance using JTC and real distance (euclidean)
    dist_est_1, _, pc1 = phase_corr_similarity(vec1, vec2, shape)
    dist_est_2, _, pc2 = phase_corr_similarity(vec1, vec3, shape)
    dist_est_3, _, pc3 = phase_corr_similarity(vec2, vec3, shape)
    dist_est_11, _, pc11 = phase_corr_similarity(vec1, vec1, shape)
    real_dist_1 = np.linalg.norm(vec1 - vec2)
    real_dist_2 = np.linalg.norm(vec1 - vec3)
    real_dist_3 = np.linalg.norm(vec2 - vec3)
    real_dist_11 = np.linalg.norm(vec1 - vec1)

    # Print the results
    print(f"Estimated distance between vec1 and vec2: {dist_est_1:.2f} pixels")
    print(f"Real distance between vec1 and vec2: {real_dist_1:.2f} pixels")
    print(f"Correlation peak: {pc1:.2f}")

    print(f"Estimated distance between vec1 and vec3: {dist_est_2:.2f} pixels")
    print(f"Real distance between vec1 and vec3: {real_dist_2:.2f} pixels")
    print(f"Correlation peak: {pc2:.2f}")

    print(f"Estimated distance between vec1 and vec3: {dist_est_3:.2f} pixels")
    print(f"Real distance between vec1 and vec3: {real_dist_3:.2f} pixels")
    print(f"Correlation peak: {pc3:.2f}")

    print(f"Estimated distance between vec1 and vec1: {dist_est_11:.2f} pixels")
    print(f"Real distance between vec1 and vec1: {real_dist_11:.2f} pixels")
    print(f"Correlation peak: {pc11:.2f}")

    # Do the same for binary JTC
    dist_est_1_bin, _, _ = vlc_similarity(img1, img2, shape)
    dist_est_2_bin, _, _ = vlc_similarity(img1, img3, shape)
    dist_est_3_bin, _, _ = vlc_similarity(img2, img3, shape)
    dist_est_11_bin, _, _ = vlc_similarity(img1, img1, shape)
    real_dist_1_bin = np.linalg.norm(img1 - img2)
    real_dist_2_bin = np.linalg.norm(img1 - img3)
    real_dist_3_bin = np.linalg.norm(img2 - img3)
    real_dist_11_bin = np.linalg.norm(img1 - img1)

    # Print the results
    print(
        f"Estimated distance (binary) between vec1 and vec2: {dist_est_1_bin:.2f} pixels"
    )
    print(f"Real distance (binary) between vec1 and vec2: {real_dist_1_bin:.2f} pixels")

    print(
        f"Estimated distance (binary) between vec1 and vec3: {dist_est_2_bin:.2f} pixels"
    )
    print(f"Real distance (binary) between vec1 and vec3: {real_dist_2_bin:.2f} pixels")

    print(
        f"Estimated distance (binary) between vec1 and vec3: {dist_est_3_bin:.2f} pixels"
    )
    print(f"Real distance (binary) between vec1 and vec3: {real_dist_3_bin:.2f} pixels")

    print(
        f"Estimated distance (binary) between vec1 and vec1: {dist_est_11_bin:.2f} pixels"
    )
    print(
        f"Real distance (binary) between vec1 and vec1: {real_dist_11_bin:.2f} pixels"
    )
