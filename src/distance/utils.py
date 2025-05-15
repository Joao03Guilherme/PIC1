import numpy as np


def binarize(img, threshold="median"):
    """
    Calculates the median pixel value of the image and binarizes it.
    - img: 2D numpy array representing the image.
    -  Method to calculate the threshold. Currently only "median" and numeric values are supported.
    Returns: 2D numpy array with binary values (-1 or 1).
    """

    if threshold == "median":
        treshold_value = np.median(img)

    else:
        if isinstance(threshold, (int, float)):
            treshold_value = threshold
        else:
            raise ValueError(
                "Invalid threshold value. Use 'median' or a numeric value."
            )

    # Binarize the image: -1 for pixels below median, 1 for pixels above
    binary_img = np.where(img < treshold_value, -1, 1)

    return binary_img


# ──────────────────────────────────────────────────────────────────────────────
# helper: mask out the zero-order patch and locate the true JTC peak
# ──────────────────────────────────────────────────────────────────────────────
def _peak_and_shift(corr_plane, img_shape):
    """
    corr_plane : real-valued array shape (H,2W) after fftshift
    img_shape  : original image shape (H,W)

    Returns
        peak_val     : amplitude of strongest off-axis peak
        (dy, dx_real): shift of img2 w.r.t img1 (same convention as phase_corr)
    """
    H, W = img_shape
    cy, cx = corr_plane.shape[0] // 2, corr_plane.shape[1] // 2

    # mask: kill central region (size of one image) that contains zero-order & autos
    mask = np.ones_like(corr_plane, dtype=bool)
    mask[cy - H // 2 : cy + H // 2 + 1, cx - W // 2 : cx + W // 2 + 1] = False

    masked = np.where(mask, corr_plane, -np.inf)
    py, px = np.unravel_index(masked.argmax(), masked.shape)
    peak_val = corr_plane[py, px]

    # raw shift from array centre
    dy = py - cy
    dx = px - cx

    # subtract the expected ±W offset of the cross-correlation lobes
    dx -= np.sign(dx) * W  # now dx is the real shift between the two images

    # unwrap circular shifts to range (-H/2,H/2], (-W/2,W/2]
    if dy > H // 2:
        dy -= H
    if dy <= -H // 2:
        dy += H
    if dx > W // 2:
        dx -= W
    if dx <= -W // 2:
        dx += W
    return peak_val, (dy, dx)
