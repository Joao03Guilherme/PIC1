#!/usr/bin/env python3
"""
binary_jtc_lightpipes_checkerboard_plot.py
------------------------------------------
Same JTC as before, now returning and plotting:

  1. Input pattern sent to the SLM (after optional checkerboard)
  2. Binarised Joint-Power-Spectrum phase map sent back to the SLM
  3. Correlation plane cropped to the camera field of view
"""
from __future__ import annotations
from LightPipes import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.data import get_test_data

EPS = 1e-6

# ───────────── geometry & optics ─────────────
slm_res      = (1920, 1200)
active_area  = (15.42e-3, 9.66e-3)
pixel_pitch  = 8e-6
fill_factor  = 0.89

cam_res      = (1280, 1024)
cam_area     = (4.608e-3, 3.686e-3)

wavelength   = 845e-9
f_len        = 75e-3
N            = 2048

slm_size     = active_area[0]
beam_waist   = 1.1 * slm_size

# ───────────── SLM pixel aperture mask ───────
def make_slm_mask(shape, open_frac):
    rows, cols = shape
    x = np.linspace(-slm_size/2, slm_size/2, cols, endpoint=False)
    y = np.linspace(-slm_size/2, slm_size/2, rows, endpoint=False)
    X, Y = np.meshgrid(x, y)
    gx = np.mod(X + pixel_pitch/2, pixel_pitch) < open_frac * pixel_pitch
    gy = np.mod(Y + pixel_pitch/2, pixel_pitch) < open_frac * pixel_pitch
    mask = gx & gy & (np.abs(Y) <= active_area[1] / 2)
    return mask.astype(float)

MASK = make_slm_mask((N, N), fill_factor)

# ───────────── checkerboard helper ───────────
def checkerboard(shape: tuple[int, int]) -> np.ndarray:
    """
    Return a +1 / -1 checkerboard mask with one-pixel pitch.
    """
    idx_sum = np.indices(shape).sum(axis=0)   # integer array
    return 1.0 - 2.0 * (idx_sum % 2)          # 0 -> +1, 1 -> -1

# ───────────── crop slices for camera FOV ────
def cam_slices():
    pitch_ft = wavelength * f_len / slm_size   # m per sim pixel
    nx = int(round(cam_area[0] / pitch_ft))
    ny = int(round(cam_area[1] / pitch_ft))
    ctr = N // 2
    return slice(ctr - ny//2, ctr + ny//2), slice(ctr - nx//2, ctr + nx//2)

SL_y, SL_x = cam_slices()

# ───────────── input-canvas helper ───────────
def mk_joint_plane(ref, obj, slm_shape, scale=0.05, binarize=True):
    rows, cols = slm_shape
    scale_255 = lambda a: np.zeros_like(a, np.uint8) if a.max()==0 else (a/a.max()*255).astype(np.uint8)
    ref255, obj255 = scale_255(ref), scale_255(obj)
    combo = np.hstack((ref255, obj255))             # 28×56
    h0, w0 = combo.shape
    fac = min(rows/h0, cols/w0) * scale
    w_new, h_new = max(1, int(w0*fac)), max(1, int(h0*fac))
    combo_rs = Image.fromarray(combo).resize((w_new, h_new), Image.BICUBIC)

    canvas = np.zeros(slm_shape, float)
    y0, x0 = (rows-h_new)//2, (cols-w_new)//2
    canvas[y0:y0+h_new, x0:x0+w_new] = np.asarray(combo_rs, float)
    if binarize:
        thr = np.median(canvas)
        return np.where(canvas > thr, 1., -1.)
    return (canvas - canvas.min()) / (canvas.ptp() + EPS)

# ───────────── JTC engine ───────────
def jtc(a0, binary_in=True, binary_jps=True, block_frac=0.01):
    # phase map for first pass
    phase_in = ((a0+1)/2*np.pi if binary_in else a0*np.pi)

    F1 = Begin(slm_size, wavelength, N)
    F1 = GaussBeam(F1, beam_waist); F1 = MultIntensity(F1, MASK)
    F1 = MultPhase(F1, phase_in)
    F1 = Lens(F1, f_len);           F1 = Forvard(F1, f_len)
    JPS = Intensity(F1, 0)

    JPS_cam = JPS[SL_y, SL_x]
    if binary_jps:
        thr = np.median(JPS_cam)
        phase_cam = np.where(JPS_cam > thr, 1., -1.)
        phase_cam = (phase_cam + 1) / 2 * np.pi
    else:
        phase_cam = (JPS_cam / JPS_cam.max()) * np.pi if JPS_cam.max() > 0 else JPS_cam

    # pad to full field for second pass
    phase_full = np.zeros_like(JPS)
    phase_full[SL_y, SL_x] = phase_cam

    F2 = Begin(slm_size, wavelength, N)
    F2 = GaussBeam(F2, beam_waist); F2 = MultIntensity(F2, MASK)
    F2 = MultPhase(F2, phase_full)
    F2 = Lens(F2, f_len);           F2 = Forvard(F2, f_len)
    Corr = Intensity(F2, 0)

    Corr_cam = Corr[SL_y, SL_x]
    h, w = Corr_cam.shape; cy, cx = h//2, w//2
    dc_val = Corr_cam.max()

    blocked = Corr_cam.copy()
    half = int(min(h, w)*block_frac)
    blocked[cy-half:cy+half+1, cx-half:cx+half+1] = 0
    peak_val = blocked.max()

    return peak_val, dc_val, phase_full, Corr_cam

# ───────────── CLI & main ───────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot input, JPS mask, and correlation plane")
    ap.add_argument("--checkerboard", action="store_true", help="apply +/-1 checkerboard")
    ap.add_argument("--binary-input", action="store_true", default=True)
    ap.add_argument("--analog-input", dest="binary_input", action="store_false")
    ap.add_argument("--binary-jps", action="store_true", default=True)
    ap.add_argument("--analog-jps", dest="binary_jps", action="store_false")
    ap.add_argument("--ref-digit", type=int, default=1)
    ap.add_argument("--obj-digit", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--dataset", choices=["mnist","fashion"], default="mnist")
    args = ap.parse_args()

    X, y = get_test_data(dataset=args.dataset)
    ref = X[y==args.ref_digit][0].reshape(28,28)/255.
    obj = X[y==args.obj_digit][0].reshape(28,28)/255.

    a0 = mk_joint_plane(ref, obj, (N,N), scale=args.scale, binarize=args.binary_input)
    if args.checkerboard:
        a0 *= checkerboard((N, N))

    peak, dc_val, phase_full, Corr_cam = jtc(
        a0,
        binary_in=args.binary_input,
        binary_jps=args.binary_jps,
        block_frac=0.01
    )

    # build images for display
    input_disp = ((a0 + 1)/2) if args.binary_input else (a0 - a0.min()) / (a0.ptp() + EPS)
    jps_disp   = phase_full / np.pi            # shows 0 or 1 for binarised
    corr_disp  = Corr_cam / Corr_cam.max()

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(input_disp * MASK, cmap='gray')
    ax[0].set_title("Input sent to SLM")
    ax[0].axis('off')

    ax[1].imshow(jps_disp, cmap='gray')
    ax[1].set_title("Binarised JPS phase mask")
    ax[1].axis('off')

    im = ax[2].imshow(corr_disp, cmap='hot')
    ax[2].set_title("Correlation plane (camera FOV)")
    ax[2].axis('off'); fig.colorbar(im, ax=ax[2])
    plt.tight_layout(); plt.show()

    print(f"peak = {peak:.3e},  DC = {dc_val:.3e},  norm = {peak/(dc_val+EPS):.4f}")
