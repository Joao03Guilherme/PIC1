
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

###############################################################################
#                               I/O helpers                                   #
###############################################################################

def load_grayscale(path: Path) -> np.ndarray:
    """Load *path* as float64 grayscale image scaled to [0, 1]."""
    img = Image.open(path).convert("L")  # 8‑bit luminance
    arr = np.asarray(img, dtype=np.float64)
    return arr / 255.0  # normalise

###############################################################################
#                         Core analytical functions                            #
###############################################################################

def smooth(img: np.ndarray, size: int = 11) -> np.ndarray:
    """Return *img* after a uniform (mean) filter of the given square *size*."""
    return ndi.uniform_filter(img, size=size, mode="nearest")


def find_peak(smoothed: np.ndarray) -> Tuple[int, int, float]:
    """Return (row, col, peak_value) in *smoothed* image."""
    idx = np.argmax(smoothed)
    peak_val = smoothed.flat[idx]
    return (*np.unravel_index(idx, smoothed.shape), peak_val)


def half_max_mask(smoothed: np.ndarray, peak_val: float) -> np.ndarray:
    """Mask of pixels ≥ 0.5·*peak_val* in *smoothed*."""
    return smoothed >= 0.5 * peak_val


def fwhm_from_mask(mask: np.ndarray) -> Tuple[int, int]:
    """Compute FWHM (rows, cols) as the bounding box of the largest component in *mask*."""
    labels, num = ndi.label(mask)
    if num == 0:
        raise RuntimeError("No pixels ≥ 50 % of peak – can’t compute FWHM.")
    sizes = ndi.sum(mask, labels, index=np.arange(1, num + 1))
    main_label = int(np.argmax(sizes) + 1)
    peak_mask = labels == main_label
    rows_any = np.any(peak_mask, axis=1)
    cols_any = np.any(peak_mask, axis=0)
    r_inds = np.where(rows_any)[0]
    c_inds = np.where(cols_any)[0]
    fwhm_rows = r_inds[-1] - r_inds[0] + 1
    fwhm_cols = c_inds[-1] - c_inds[0] + 1
    return fwhm_rows, fwhm_cols


def rms_noise(original: np.ndarray, signal_mask: np.ndarray) -> float:
    """RMS of *original* pixels **outside** *signal_mask*."""
    noise_pixels = original[~signal_mask]
    return np.sqrt(np.mean(np.square(noise_pixels)))


def snr_linear(peak_val: float, rms: float) -> float:
    return np.inf if rms == 0 else peak_val / rms


def analyse_image(path: Path, smooth_size: int = 11) -> Dict[str, float]:
    """Run full analysis for a single PNG file."""
    original = load_grayscale(path)
    smoothed = smooth(original, size=smooth_size)

    pr, pc, peak_val_smooth = find_peak(smoothed)
    # Use unsmoothed amplitude at the peak coordinates for the SNR numerator
    peak_val_raw = original[pr, pc]

    mask = half_max_mask(smoothed, peak_val_smooth)  # 50 % region from smoothed image
    fwhm_r, fwhm_c = fwhm_from_mask(mask)

    rms_n = rms_noise(original, mask)
    snr_lin = snr_linear(peak_val_raw, rms_n)
    snr_db = 20 * np.log10(snr_lin) if snr_lin > 0 else -np.inf

    return {
        "file": path.name,
        "peak_value": peak_val_raw,
        "fwhm_rows": fwhm_r,
        "fwhm_cols": fwhm_c,
        "rms_noise": rms_n,
        "snr_linear": snr_lin,
        "snr_db": snr_db,
    }

###############################################################################
#                                    CLI                                      #
###############################################################################

def main():
    if len(sys.argv) > 1:
        candidates = [Path(p) for p in sys.argv[1:]]
    else:
        candidates = sorted(Path('.').glob('?v?.png'))

    if not candidates:
        print("No matching PNG files found.")
        sys.exit(1)

    print("file, peak, fwhm_rows, fwhm_cols, rms_noise, snr_linear, snr_db")
    for p in candidates:
        try:
            res = analyse_image(p)
            print(
                f"{res['file']}, {res['peak_value']:.6f}, {res['fwhm_rows']}, "
                f"{res['fwhm_cols']}, {res['rms_noise']:.6e}, {res['snr_linear']:.2f}, "
                f"{res['snr_db']:.2f}"
            )
        except Exception as exc:
            print(f"Error processing {p}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
