#!/usr/bin/env python3
"""
SLM Phase Calibration (no CLI)
==============================

Reads camera frames named `cam_mirrorXXX.ext` in `camera_captures/`,
extracts the fringe phase for each greyscale value, performs a linear
fit (phase vs. greyscale), and saves the results.

Outputs
-------
phase_calibration.npz : NumPy archive of greyscales, phases, fit params
slm_calibration.png   : Calibration plot
Console print-out     : Slope, intercept, R², std. error
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from skimage import io


# --------------------------------------------------------------------
# User-editable settings
FOLDER = "camera_captures"   # directory with cam_mirrorXXX.<ext> files
EXT    = ".png"              # image extension
PLOT   = "slm_calibration.png"
# --------------------------------------------------------------------


def load_camera_images(folder: str, ext: str):
    """Return list[image], ndarray[int] of greyscale values."""
    pattern = re.compile(r"cam_mirror(\d{3})", re.IGNORECASE)
    images, greys = [], []

    for f in sorted(Path(folder).glob(f"*{ext}")):
        m = pattern.search(f.stem)
        if not m:
            continue
        greys.append(int(m.group(1)))
        img = io.imread(f.as_posix())
        if img.ndim == 3:                     # RGB → grayscale
            img = img[..., :3].mean(axis=2)
        images.append(img.astype(float))

    greys = np.asarray(greys)
    order = np.argsort(greys)
    return [images[i] for i in order], greys[order]


def dominant_phase_1d(img: np.ndarray) -> float:
    """Global phase of dominant spatial frequency (rad)."""
    sig = img.mean(axis=0)              # average rows
    sig -= sig.mean()                   # remove DC

    fft = np.fft.fft(sig)
    freqs = np.fft.fftfreq(sig.size)

    pos = freqs > 0                     # positive freqs
    k = np.where(pos)[0][np.argmax(np.abs(fft[pos]))]

    return np.angle(fft[k])


def extract_phases(images):
    phases = np.array([dominant_phase_1d(im) for im in images])
    return np.unwrap(phases) - phases[0]    # reference to first image


def calibrate(greys, phases):
    slope, intercept, r, *_ = stats.linregress(greys, phases)
    return slope, intercept, r ** 2


def main():
    folder = Path(FOLDER)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder '{folder}' does not exist.")

    images, greys = load_camera_images(folder, EXT)
    if not images:
        raise RuntimeError(f"No images matching 'cam_mirrorXXX{EXT}' found in {folder}")

    phases = extract_phases(images)
    slope, intercept, r2 = calibrate(greys, phases)

    # Save raw data + fit
    np.savez(
        "phase_calibration.npz",
        greyscale=greys,
        phase=phases,
        slope=slope,
        intercept=intercept,
        r2=r2,
    )

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(greys, phases, label="Measured phase", zorder=3)
    xfit = np.linspace(greys.min(), greys.max(), 200)
    plt.plot(
        xfit,
        slope * xfit + intercept,
        label=fr"$\varphi = {slope:.4f}\,G + {intercept:.4f}$" "\n"
              fr"$R^2 = {r2:.4f}$",
        linewidth=2,
    )
    plt.xlabel("Mirror greyscale value $G$")
    plt.ylabel("Phase shift $\\varphi$ (rad)")
    plt.title("SLM Phase Calibration")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT, dpi=300)

    print(f"Saved calibration plot to '{PLOT}'")
    print("---------- Fit parameters ----------")
    print(f"Slope      : {slope:.6f} rad/greyscale")
    print(f"Intercept  : {intercept:.6f} rad")
    print(f"R²         : {r2:.6f}")


if __name__ == "__main__":
    main()
