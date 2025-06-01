from __future__ import annotations

from LightPipes import *           # optical propagation – NO NumPy FFTs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.data import get_test_data         # helper for MNIST/F-MNIST

EPS = 1e-6

# -------------------------- real-SLM parameters ----------------------------
slm_resolution = (1920, 1200)          # (cols, rows) – informational only
active_area    = (15.42e-3, 9.66e-3)   # physical size (width, height) [m]
pixel_pitch    = 8e-6                  # 8 µm  (≈ active_area[0]/1920)
fill_factor    = 0.89                 # 92 % duty cycle (open area)

# ---------------------- optical / simulation settings ----------------------
wavelength = 633e-9                    # 633 nm He–Ne
f_lens     = 100e-3                    # 100 mm Fourier lens (×2 passes)
N          = 2048                      # simulation grid (square)

slm_size   = active_area[0]            # use SLM width for LightPipes window
beam_waist = 1.3 * slm_size            # 

# binary threshold for digits (0-255); None → median of canvas
img_thresh = None

# ---------------------------------------------------------------------------
# Helper: build pixel-grid amplitude mask (returns values ∈ {0,1})
# ---------------------------------------------------------------------------

def make_slm_aperture(grid_shape: tuple[int, int],
                      slm_shape: tuple[int, int],
                      fill_factor: float = 0.88) -> np.ndarray:
    """Return a binary mask representing the SLM pixel aperture.

    A value of 1 means light is transmitted by the pixel; 0 represents the
    inter-pixel gap (dark metal).  The mask is centred on the simulation
    window, which is square (size = *slm_size*).  Pixels outside the real SLM
    active area are set to 0 (black frame).
    """
    Ny, Nx = grid_shape                # LightPipes: (rows, cols)
    cols_slm, rows_slm = slm_shape

    # Generate physical x/y coordinates for the simulation grid (metres)
    x = np.linspace(-slm_size/2, slm_size/2, Nx, endpoint=False)
    y = np.linspace(-slm_size/2, slm_size/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Pixel-grid open regions (duty cycle = *fill_factor*)
    gx = (np.mod(X + pixel_pitch/2, pixel_pitch) < fill_factor*pixel_pitch)
    gy = (np.mod(Y + pixel_pitch/2, pixel_pitch) < fill_factor*pixel_pitch)
    pix_open = gx & gy

    # Mask out area outside the active height (since window is square)
    half_h = active_area[1] / 2
    within_height = np.abs(Y) <= half_h

    return (pix_open & within_height).astype(float)


# Build the global SLM amplitude mask once ----------------------------------
MASK = make_slm_aperture((N, N), slm_shape=slm_resolution, fill_factor=fill_factor)

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
# correlation engine (only two insertions marked «NEW»)
# ---------------------------------------------------------------------------

def perform_jtc_correlation(a0_pair,
                            binary_input: bool = True,
                            binary_jps: bool   = True,
                            blocking_factor: float = 0.1):
    """Two-pass Binary JTC – identical logic, but now with:
       • TEM₀₀ Gaussian illumination (waist = *beam_waist*)
       • physical SLM pixel mask (*MASK*) acting as an amplitude pupil
    """

    # ---- phase encoding of the joint input ---------------------------------
    phase_pair = (a0_pair + 1)/2*np.pi if binary_input else a0_pair*np.pi

    # ---- first optical pass (JPS) ------------------------------------------
    F1 = Begin(slm_size, wavelength, N)
    F1 = GaussBeam(F1, beam_waist)          # «NEW» Gaussian envelope
    F1 = MultIntensity(F1, MASK)            # pixel-grid aperture (amp.)
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
    F2 = GaussBeam(F2, beam_waist)          # 
    F2 = MultIntensity(F2, MASK)            # pixel-grid aperture (amp.)
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

    peak_val = corr_masked[600:1400, 600:1400].max()

    return peak_val, central_dc, Corr_int, corr_masked

# ------------------------- main script ------------------------------------
if __name__ == "__main__":

    cli = argparse.ArgumentParser(
        description="Binary / Analogue Joint-Transform Correlator simulation")
    cli.add_argument("--binary-input",  action="store_true", default=True)
    cli.add_argument("--analog-input",  dest="binary_input", action="store_false")
    cli.add_argument("--binary-jps",    action="store_true", default=True)
    cli.add_argument("--analog-jps",    dest="binary_jps",   action="store_false")
    cli.add_argument("--ref-digit",     type=int, default=1)
    cli.add_argument("--obj-digit",     type=int, default=2)
    cli.add_argument("--scale",         type=float, default=1)
    cli.add_argument("--dataset",       choices=["mnist", "fashion"], default="mnist")
    args = cli.parse_args()

    # ------------------- prepare joint input --------------------------------
    X, y = get_test_data(dataset=args.dataset)
    ref = X[y == args.ref_digit][0].reshape(28, 28) / 255.
    obj = X[y == args.obj_digit][0].reshape(28, 28) / 255.

    a0 = create_joint_input_plane(ref, obj, (N, N), img_thresh,
                                  display_scale_factor=args.scale,
                                  binarize=args.binary_input)

    # ------------------- run correlator -------------------------------------
    peak_val, central_dc, Corr_int, corr_masked = perform_jtc_correlation(
        a0,
        binary_input=args.binary_input,
        binary_jps=args.binary_jps,
        blocking_factor=0.005
    )

    # ------------------- plots ---------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.imshow(Corr_int[950:1100, 800:1250], cmap='hot')
    plt.title("Correlation Plane")
    plt.colorbar(label='Intensity')
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.tight_layout()
    plt.show()

    print(f"peak = {peak_val:.3e},  DC = {central_dc:.3e},  "
          f"norm = {peak_val/(central_dc+EPS):.4f}")
    
