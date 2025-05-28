#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Binary Joint-Transform Correlator (single-SLM) – LightPipes simulation
# ---------------------------------------------------------------------------
#  Implements:  B. Javidi & J. L. Horner, “Single spatial light modulator joint
#  transform correlator”, Appl. Opt. 28 (5): 1027 – 1032, 1989.
#
#  THIS VERSION **adds the real SLM geometry only**:
#      • resolution ............ 1920 × 1200 px  (informative)
#      • active area ........... 15.42 mm × 9.66 mm
#      • pixel pitch (physical)  8 µm            → gives the same 15.42 mm width
#      • fill factor ........... 0.92            (grid helper kept, but unused)
#
#  Everything else – phase mapping, two-pass JTC, peak search, plots – is
#  **bit-for-bit identical** to the reference script you provided.
# ---------------------------------------------------------------------------

from LightPipes import *           # optical propagation only – NO NumPy FFTs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.data import get_test_data, get_dataset         # helper for MNIST/F-MNIST

EPS = 1e-6

# -------------------------- real-SLM parameters ----------------------------
slm_resolution = (1920, 1200)          # pixels (Wx, Wy) – for reference only
active_area    = (15.42e-3, 9.66e-3)   # physical size (width, height)  [m]
pixel_pitch    = 8e-6                  # 8 µm  (≈ active_area[0]/1920)
fill_factor    = 0.92                  # 92 % duty (grid helper below)

# ---------------------- optical / simulation settings ----------------------
wavelength = 633e-9                    # 633 nm He-Ne
f_lens     = 100e-3                    # 100 mm Fourier lens
N          = 512                       # simulation grid (square) – unchanged
# use the **width** of the real SLM for LightPipes’ square window:
slm_size   = active_area[0]            # 15.42 mm
# binary threshold for digits (0-255); None → median
img_thresh = None

# ---------------------------------------------------------------------------
# (unchanged) correlation engine
# ---------------------------------------------------------------------------
def perform_jtc_correlation(a0_pair, binary_input=True, binary_jps=True,
                            blocking_factor=0.1, digit1=None, digit2=None):
    """
    EXACTLY the same function you supplied. No functional edits were made.
    Only comments trimmed for brevity.
    """
    # ---- phase encoding of the joint input ---------------------------------
    phase_pair = (a0_pair + 1)/2*np.pi if binary_input else a0_pair*np.pi

    # ---- first optical pass (JPS) ------------------------------------------
    F1 = Begin(slm_size, wavelength, N)
    F1 = MultPhase(F1, phase_pair)
    F1 = Lens(F1, f_lens)
    F1 = Forvard(F1, f_lens)
    JPS_int = Intensity(F1, 0)

    # ---- JPS processing / binarisation -------------------------------------
    if binary_jps:
        thr_JPS   = np.median(JPS_int)
        JPS_bin   = np.where(JPS_int > thr_JPS, 1., -1.)
        phase_JPS = (JPS_bin + 1)/2*np.pi
    else:
        JPS_norm  = JPS_int/JPS_int.max() if JPS_int.max() > 0 else JPS_int
        phase_JPS = JPS_norm*np.pi

    # ---- second optical pass (correlation) ---------------------------------
    F2 = Begin(slm_size, wavelength, N)
    F2 = MultPhase(F2, phase_JPS)
    F2 = Lens(F2, f_lens)
    F2 = Forvard(F2, f_lens)
    Corr_int = Intensity(F2, 0)

    # ---- peak search (unchanged) -------------------------------------------
    center = N//2
    central_dc = Corr_int.max()

    corr_masked = Corr_int.copy()
    half = int(N*blocking_factor)
    corr_masked[center-half:center+half+1,
                center-half:center+half+1] = 0.0

    left  = corr_masked.copy(); left [:, center:] = 0
    right = corr_masked.copy(); right[:, :center] = 0
    lpk, rpk = left.max(), right.max()

    if lpk > rpk:
        peak_val = lpk
        peak_idx = np.unravel_index(np.argmax(left), left.shape)
        search_region = "left half (stronger peak)"
    else:
        peak_val = rpk
        peak_idx = np.unravel_index(np.argmax(right), right.shape)
        search_region = "right half (stronger peak)"

    dy = peak_idx[0] - center
    dx = peak_idx[1] - center
    print(f"Finding correlation peak in {search_region}: peak at ({dx}, {dy})")

    return peak_val, central_dc, (dy, dx), Corr_int, corr_masked

# ---------------------------------------------------------------------------
# fill-factor helper (kept unchanged, for completeness – mask is **not** used)
# ---------------------------------------------------------------------------
def slm_pixel_aperture(x, y, width, height, duty=fill_factor):
    gx = (np.mod(x + width/2, pixel_pitch) < duty*pixel_pitch)
    gy = (np.mod(y + height/2, pixel_pitch) < duty*pixel_pitch)
    return (gx & gy).astype(float)

# ---------------------------------------------------------------------------
# helper: create the joint input plane (unchanged)
# ---------------------------------------------------------------------------
def create_joint_input_plane(digit_array_ref: np.ndarray,
                             digit_array_obj: np.ndarray,
                             slm_shape: tuple[int, int],
                             thresh,
                             display_scale_factor: float = 0.2,
                             binarize: bool = True):

    slm_rows, slm_cols = slm_shape

    def scale_digit_to_255(d):
        m = d.max()
        return np.zeros_like(d, np.uint8) if m == 0 else (d/m*255).astype(np.uint8)

    ref_255 = scale_digit_to_255(digit_array_ref)
    obj_255 = scale_digit_to_255(digit_array_obj)

    combo = np.hstack((ref_255, obj_255))       # 28 × 56
    H0, W0 = combo.shape

    scale = min(slm_rows/H0, slm_cols/W0)
    Wf = int(W0*scale*display_scale_factor)
    Hf = int(H0*scale*display_scale_factor)
    Wf, Hf = max(1, Wf), max(1, Hf)

    combo_rs = Image.fromarray(combo).resize((Wf, Hf), Image.BICUBIC)
    canvas   = np.zeros(slm_shape, float)
    y0, x0   = (slm_rows-Hf)//2, (slm_cols-Wf)//2
    canvas[y0:y0+Hf, x0:x0+Wf] = np.asarray(combo_rs, float)

    if binarize:
        t = np.median(canvas) if thresh is None else thresh
        return np.where(canvas > t, 1., -1.)
    return (canvas - canvas.min())/(canvas.ptp() + EPS)

# ---------------------------------------------------------------------------
# everything below is IDENTICAL to your reference script
# ---------------------------------------------------------------------------
# build a default joint-input plane for the “demo run”
X, y = get_test_data(dataset="mnist")
ref_idx = np.where(y == 1)[0][0]
ref_digit_array = X[ref_idx].reshape(28,28)/255.0
obj_digit_array = ref_digit_array.copy()                      # autocorrelation
a0 = create_joint_input_plane(ref_digit_array,
                              obj_digit_array,
                              (N, N),
                              img_thresh,
                              display_scale_factor=0.20)

# ------------------------- main script ------------------------------------
if __name__ == "__main__":

    cli = argparse.ArgumentParser(
        description="Binary / Analogue Joint-Transform Correlator simulation")
    cli.add_argument("--binary-input",  action="store_true", default=True)
    cli.add_argument("--analog-input",  dest="binary_input", action="store_false")
    cli.add_argument("--binary-jps",    action="store_true", default=True)
    cli.add_argument("--analog-jps",    dest="binary_jps",   action="store_false")
    cli.add_argument("--ref-digit",     type=int, default=1)
    cli.add_argument("--obj-digit",     type=int, default=1)
    cli.add_argument("--scale",         type=float, default=0.2)
    cli.add_argument("--dataset",       choices=["mnist","fashion"], default="mnist")
    args = cli.parse_args()

    X, y = get_test_data(dataset=args.dataset)
    ref = X[y == args.ref_digit][0].reshape(28,28)/255.
    obj = X[y == args.obj_digit][0].reshape(28,28)/255.

    a0 = create_joint_input_plane(ref, obj, (N,N), img_thresh,
                                  display_scale_factor=args.scale,
                                  binarize=args.binary_input)

    peak_val, central_dc, (dy, dx), Corr_int, corr_masked = perform_jtc_correlation(
        a0,
        binary_input=args.binary_input,
        binary_jps=args.binary_jps,
        blocking_factor=0.05
    )

    # --------- recompute JPS just for display (same as original) ------------
    phase_pair = (a0 + 1)/2*np.pi if args.binary_input else a0*np.pi
    Ftmp = Begin(slm_size, wavelength, N)
    Ftmp = MultPhase(Ftmp, phase_pair)
    Ftmp = Lens(Ftmp, f_lens)
    Ftmp = Forvard(Ftmp, f_lens)
    JPS_int = Intensity(Ftmp, 0)

    if args.binary_jps:
        thr = np.median(JPS_int)
        jps_display = np.where(JPS_int > thr, 1., -1.)
        jps_title   = "Binary JPS (Fourier plane)"
    else:
        jps_display = JPS_int/JPS_int.max() if JPS_int.max() else JPS_int
        jps_title   = "Analog JPS (Fourier plane)"

    # ----------------------------- plots ------------------------------------
    fig, axs = plt.subplots(2,2, figsize=(13,8))

    vmin_in, vmax_in = (-1,1) if args.binary_input else (0,1)
    axs[0,0].imshow(a0, cmap='gray', vmin=vmin_in, vmax=vmax_in)
    axs[0,0].set_title('Binary input plane' if args.binary_input else 'Analog input plane')
    axs[0,0].axis('off')

    axs[0,1].imshow(jps_display,
                    cmap='gray' if args.binary_jps else 'viridis',
                    vmin=-1 if args.binary_jps else 0,
                    vmax= 1 if args.binary_jps else None)
    axs[0,1].set_title(jps_title); axs[0,1].axis('off')

    axs[1,0].imshow(Corr_int, cmap='inferno')
    axs[1,0].set_title(f'Correlation ({"Binary" if args.binary_jps else "Analog"} JPS)')
    axs[1,0].axis('off')

    axs[1,1].imshow(corr_masked, cmap='inferno')
    center = N//2
    axs[1,1].plot(center+dx, center+dy, 'wo', ms=8, mew=1.5)
    axs[1,1].set_title('Masked correlation plane'); axs[1,1].axis('off')

    plt.tight_layout(); plt.show()

    print(f"peak = {peak_val:.3e},  DC = {central_dc:.3e},  "
          f"norm = {peak_val/(central_dc+EPS):.4f},  offset = ({dy},{dx})")
