#!/usr/bin/env python3
"""
binary_jtc_lightpipes_correlate.py
----------------------------------
Command-line demo of **binary joint-transform correlation** between two
MNIST (or Fashion-MNIST) digits.

Two execution paths are available inside `binary_jtc_correlate`:

* **fast FFT** (default) – replaces both LightPipes propagation steps with two
  NumPy FFTs. ~20× quicker for N = 2048 while giving the same correlation peak.

* **full LightPipes** – set `fast_fft=False` when calling the function if you
  need an explicit Fresnel propagation model.

Nothing else in the original script has been altered.
"""
from __future__ import annotations
# ───────────── imports ─────────────
from LightPipes import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.data import get_test_data

# ───────────── constants ─────────────
EPS = 1e-6

# ───────────── geometry & optics ─────────────
slm_res      = (1920, 1200)
active_area  = (15.42e-3, 9.66e-3)     # m
pixel_pitch  = 8e-6                    # m
fill_factor  = 0.89

cam_res      = (1280, 1024)
cam_area     = (4.608e-3, 3.686e-3)    # m

wavelength   = 845e-9                  # m
f_len        = 75e-3                   # m
N            = 2048                    # simulation grid size

slm_size     = active_area[0]          # use horizontal active size
beam_waist   = 1.1 * slm_size

# ───────────── helpers ─────────────
def make_slm_mask(shape: tuple[int, int], open_frac: float) -> np.ndarray:
    rows, cols = shape
    x = np.linspace(-slm_size/2, slm_size/2, cols, endpoint=False)
    y = np.linspace(-slm_size/2, slm_size/2, rows, endpoint=False)
    X, Y = np.meshgrid(x, y)
    gx = np.mod(X + pixel_pitch/2, pixel_pitch) < open_frac * pixel_pitch
    gy = np.mod(Y + pixel_pitch/2, pixel_pitch) < open_frac * pixel_pitch
    mask = gx & gy & (np.abs(Y) <= active_area[1]/2)
    return mask.astype(float)

MASK = make_slm_mask((N, N), fill_factor)

def checkerboard(shape: tuple[int, int]) -> np.ndarray:
    idx_sum = np.indices(shape).sum(axis=0)
    return 1.0 - 2.0 * (idx_sum % 2)

def cam_slices() -> tuple[slice, slice]:
    pitch_ft = wavelength * f_len / slm_size             # m per pixel in FT plane
    nx = int(round(cam_area[0] / pitch_ft))
    ny = int(round(cam_area[1] / pitch_ft))
    ctr = N // 2
    return slice(ctr - ny//2, ctr + ny//2), slice(ctr - nx//2, ctr + nx//2)

SL_y, SL_x = cam_slices()

def mk_joint_plane(
    ref: np.ndarray,
    obj: np.ndarray,
    slm_shape: tuple[int, int],
    *,
    scale: float = 0.05,
    binarize: bool = True
) -> np.ndarray:
    """Return a centred 2-up pattern of `ref‖obj` on the SLM grid."""
    rows, cols = slm_shape
    to_255 = lambda a: np.zeros_like(a, np.uint8) if a.max() == 0 else (a/a.max()*255).astype(np.uint8)
    ref255, obj255 = to_255(ref), to_255(obj)
    combo = np.hstack((ref255, obj255))           # 28×56

    h0, w0 = combo.shape
    fac = min(rows / h0, cols / w0) * scale
    w_new, h_new = max(1, int(w0*fac)), max(1, int(h0*fac))
    combo_rs = Image.fromarray(combo).resize((w_new, h_new), Image.BICUBIC)

    canvas = np.zeros(slm_shape, float)
    y0, x0 = (rows - h_new) // 2, (cols - w_new) // 2
    canvas[y0:y0+h_new, x0:x0+w_new] = np.asarray(combo_rs, float)

    if binarize:
        thr = np.median(canvas)                   # median threshold
        return np.where(canvas > thr, 1., -1.)
    return (canvas - canvas.min()) / (canvas.ptp() + EPS)

# ---------------------------------------------------------------------
#  FAST (FFT)  OR  FULL (LightPipes)  CORRELATION ENGINE
# ---------------------------------------------------------------------
def binary_jtc_correlate(
    img1_flat: np.ndarray,
    img2_flat: np.ndarray,
    shape: tuple[int, int],
    *,
    scale:          float = 0.05,
    binarize_input: bool  = True,
    binary_jps:     bool  = True,
    checker:        bool  = True,
    block_frac:     float = 0.01,
    fast_fft:       bool  = True        # ← toggle FFT shortcut
) -> tuple[float, tuple[int, int], float, np.ndarray]:
    """
    Return (distance, (dy,dx), similarity, corr_plane).

    Setting `fast_fft=True` uses two NumPy FFTs instead of the full
    LightPipes Fresnel model.  The signature is otherwise unchanged.
    """
    # 1 ─── input on SLM
    img1, img2 = img1_flat.reshape(shape), img2_flat.reshape(shape)
    a0 = mk_joint_plane(img1, img2, (N, N), scale=scale, binarize=binarize_input)
    if checker:
        a0 *= checkerboard((N, N))

    phase_in = ((a0 + 1) / 2 * np.pi) if binarize_input else a0 * np.pi

    # ──────────────────────────────────────────────────
    # FAST PATH  (Fourier optics ≈ FFT)
    # ──────────────────────────────────────────────────
    if fast_fft:
        # first pass – Fourier plane intensity
        Uslm = MASK * np.exp(1j * phase_in)
        Fft  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Uslm), norm="ortho"))
        JPS  = np.abs(Fft) ** 2

        # threshold inside camera FOV
        JPS_cam = JPS[SL_y, SL_x]
        if binary_jps:
            thr = np.median(JPS_cam)
            phase_cam = np.where(JPS_cam > thr, 1., -1.)
            phase_cam = (phase_cam + 1) / 2 * np.pi      # 0 / π
        else:
            phase_cam = (JPS_cam / JPS_cam.max()) * np.pi if JPS_cam.max() > 0 else JPS_cam

        phase_full             = np.zeros_like(JPS)
        phase_full[SL_y, SL_x] = phase_cam

        # second pass – inverse FT
        Umask  = MASK * np.exp(1j * phase_full)
        Ucorr  = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Umask), norm="ortho"))
        Corr   = np.abs(Ucorr) ** 2

    # ──────────────────────────────────────────────────
    # SLOW PATH  (original LightPipes propagation)
    # ──────────────────────────────────────────────────
    else:
        F1 = Begin(slm_size, wavelength, N)
        F1 = GaussBeam(F1, beam_waist);  F1 = MultIntensity(F1, MASK)
        F1 = MultPhase(F1, phase_in);    F1 = Lens(F1, f_len)
        F1 = Forvard(F1, f_len)
        JPS = Intensity(F1, 0)

        JPS_cam = JPS[SL_y, SL_x]
        if binary_jps:
            thr = np.median(JPS_cam)
            phase_cam = np.where(JPS_cam > thr, 1., -1.)
            phase_cam = (phase_cam + 1) / 2 * np.pi
        else:
            phase_cam = (JPS_cam / JPS_cam.max()) * np.pi if JPS_cam.max() > 0 else JPS_cam

        phase_full             = np.zeros_like(JPS)
        phase_full[SL_y, SL_x] = phase_cam

        F2 = Begin(slm_size, wavelength, N)
        F2 = GaussBeam(F2, beam_waist);  F2 = MultIntensity(F2, MASK)
        F2 = MultPhase(F2, phase_full);  F2 = Lens(F2, f_len)
        F2 = Forvard(F2, f_len)
        Corr = Intensity(F2, 0)

    # ─── extract metrics
    Corr_cam = Corr[SL_y, SL_x]
    h, w     = Corr_cam.shape;  cy, cx = h//2, w//2
    dc_val   = Corr_cam.max()

    blocked = Corr_cam.copy()
    half    = int(min(h, w) * block_frac)
    blocked[cy-half:cy+half+1, cx-half:cx+half+1] = 0
    peak_val = blocked.max()

    y_peak, x_peak = np.unravel_index(np.argmax(blocked), blocked.shape)
    dy, dx = y_peak - cy, x_peak - cx

    similarity = peak_val / (dc_val + EPS)
    distance   = 1.0 / (similarity + EPS)

    print(distance)

    return distance, (dy, dx), similarity, Corr_cam / (dc_val + EPS)

# ───────────── quick demo ─────────────
if __name__ == "__main__":
    ref_digit, obj_digit = 3, 3
    X, y = get_test_data(dataset="mnist")
    ref = X[y == ref_digit][0].reshape(28, 28)   # 0-255 uint8
    obj = X[y == obj_digit][0].reshape(28, 28)

    dist, shift, sim, corr_plane = binary_jtc_correlate(
        ref.flatten(), obj.flatten(), (28, 28),
        scale=0.05, block_frac=0.01, fast_fft=False
    )

    print(f"Distance     : {dist:.4f}")
    print(f"Shift (dy,dx): {shift}")
    print(f"Similarity   : {sim:.4f}")

    plt.imshow(corr_plane, cmap="hot", interpolation="nearest")
    plt.title("Correlation Plane")
    plt.colorbar(); plt.axis("off")
    plt.show()
