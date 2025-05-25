# Binary Joint Transform Correlator (single‑SLM) simulation using LightPipes
# ---------------------------------------------------------------------------
# Implements the architecture described in:
#   B. Javidi & J. L. Horner, “Single spatial light modulator joint transform
#   correlator”, Appl. Opt. 28(5):1027–1032, 1989.
#
# Pipeline
# --------
# 1.  Read a **reference** image and a **test/object** image, threshold each to a
#     binary (+1/‑1) pattern, and place them **side‑by‑side** on a single SLM
#     plane (reference on the left half, test on the right half).
# 2.  Use a Fourier lens (LightPipes `Lens` → `Forvard`) to obtain the **joint
#     power spectrum** (JPS) at the Fourier plane.
# 3.  Threshold that JPS to a binary (+1/‑1) pattern and write it back onto the
#     *same* SLM (second exposure).
# 4.  Apply the lens once more to form the **correlation plane**; the resulting
#     intensity contains autocorrelation peaks at ±2x₀ and the desired
#     cross‑correlation peak at the origin if the object matches the reference.
# 5.  Display the binary input plane, the binary JPS, and the final correlation
#     intensity.
#
# **Important**
# * Uses LightPipes’ *physical* propagation instead of any NumPy FFTs.
# * Replace `reference.png` and `object.png` with your own 8‑bit grayscale
#   images. They will be resized automatically to **(N × N/2)** so that together
#   they fill the **N × N** SLM aperture.
# * Dependencies: `pip install LightPipes matplotlib pillow numpy`.
# ---------------------------------------------------------------------------

from LightPipes import *          # optical propagation library (no FFT calls)
import numpy as np                # array handling only (no FFT)
from PIL import Image             # image I/O
import matplotlib.pyplot as plt   # visualisation
import sys
from pathlib import Path

# Add parent directory to path to import the data module
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.data import get_test_data, get_dataset  # For loading MNIST/Fashion MNIST digits

# -------------------------- user‑adjustable parameters ----------------------
wavelength = 633e-9               # 633 nm He‑Ne laser
slm_size   = 6e-3                 # 6 mm × 6 mm SLM aperture
N          = 512                  # simulation grid (N × N)
f_lens     = 100e-3               # 100 mm focal‑length Fourier lens
pixel_pitch = slm_size / N        # SLM pixel pitch (physical size of each pixel)

# binary threshold for images (0–255); set to None to use the pixel median
img_thresh = None

# ------------------------------------------------------------------
# SLM fill-factor amplitude mask
# ------------------------------------------------------------------
def slm_pixel_aperture(x, y, width, height, duty=0.93):
    """1 on each pixel face, 0 in the gaps (duty<1)."""
    gx = (np.mod(x + width/2, pixel_pitch) < duty*pixel_pitch)
    gy = (np.mod(y + height/2, pixel_pitch) < duty*pixel_pitch)
    return (gx & gy).astype(float)

# ---------------------------------------------------------------------------
# helper functions

def create_joint_input_plane(digit_array_ref: np.ndarray, digit_array_obj: np.ndarray, 
                             slm_shape: tuple[int, int], thresh, 
                             display_scale_factor: float = 0.4,
                             binarize: bool = True):
    """
    Creates a joint input plane for the SLM.
    The two input digit arrays are placed side-by-side, then this combined image
    is scaled (preserving aspect ratio) to fit the SLM, further scaled by
    display_scale_factor, and centered on the SLM canvas.

    Args:
        digit_array_ref: 2D numpy array for the reference digit.
        digit_array_obj: 2D numpy array for the object/test digit.
        slm_shape: Tuple (rows, cols) for the full SLM canvas (e.g., N, N).
        thresh: Binarization threshold (0-255) or None to use median.
        display_scale_factor: Factor to scale the combined image on the SLM.
                              0.4 means 40% of the size that would snugly fit.
        binarize: Whether to binarize the output (+1/-1) or keep grayscale (0-1).
    Returns:
        A 2D numpy array (slm_shape) with values either +1/-1 if binarized or 
        normalized 0-1 if not binarized.
    """
    slm_rows, slm_cols = slm_shape

    # Scale individual digits to 0-255
    def scale_digit_to_255(digit_arr):
        max_val = digit_arr.max()
        if max_val == 0:
            return np.zeros_like(digit_arr, dtype=np.uint8)
        return (digit_arr / max_val * 255.0).astype(np.uint8)

    ref_scaled_255 = scale_digit_to_255(digit_array_ref)
    obj_scaled_255 = scale_digit_to_255(digit_array_obj)

    # Combine digits side-by-side
    combined_digits_arr_raw = np.hstack((ref_scaled_255, obj_scaled_255))
    H_comb_raw, W_comb_raw = combined_digits_arr_raw.shape

    img_pil_combined = Image.fromarray(combined_digits_arr_raw, 'L')

    # Calculate dimensions to fit combined image onto SLM, preserving aspect ratio
    scale_h_slm = slm_rows / H_comb_raw
    scale_w_slm = slm_cols / W_comb_raw
    scale_slm = min(scale_h_slm, scale_w_slm)

    fit_pil_W_on_slm = int(W_comb_raw * scale_slm)
    fit_pil_H_on_slm = int(H_comb_raw * scale_slm)

    # Apply the display_scale_factor to make the combined image smaller
    final_display_W = int(fit_pil_W_on_slm * display_scale_factor)
    final_display_H = int(fit_pil_H_on_slm * display_scale_factor)
    
    final_display_W = max(1, final_display_W) # Ensure at least 1x1
    final_display_H = max(1, final_display_H)

    # Resize the combined digit image
    img_resized_pil_combined = img_pil_combined.resize((final_display_W, final_display_H), Image.BICUBIC)
    
    # Create SLM canvas
    slm_canvas_arr_float = np.zeros(slm_shape, dtype=np.float64)
    
    # Calculate top-left position to paste the resized combined image (centered)
    paste_y = (slm_rows - final_display_H) // 2
    paste_x = (slm_cols - final_display_W) // 2
    
    resized_combined_arr = np.asarray(img_resized_pil_combined, dtype=np.float64)
    
    slm_canvas_arr_float[paste_y : paste_y + final_display_H, paste_x : paste_x + final_display_W] = resized_combined_arr
    
    if binarize:
        # Binarize the entire SLM canvas to +1/-1
        t_val = np.median(slm_canvas_arr_float) if thresh is None else thresh
        return np.where(slm_canvas_arr_float > t_val, 1.0, -1.0)
    else:
        # Normalize to 0-1 range without binarizing
        min_val = slm_canvas_arr_float.min()
        max_val = slm_canvas_arr_float.max()
        if max_val > min_val:  # Avoid division by zero
            return (slm_canvas_arr_float - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(slm_canvas_arr_float)

# -------------------------- build the joint input plane --------------------
# Load MNIST digits from data.py
X, y = get_test_data(dataset="mnist")  # Use the custom data loading function
# Select a digit "1" as reference
ref_idx = np.where(y == 1)[0][0]  # Get the first occurrence of digit "1"
# Select the same digit for object (autocorrelation example)
obj_idx = ref_idx  
# Reshape from flat 784 vector to 28x28 image
ref_digit_array = X[ref_idx].reshape(28, 28) / 255.0  # Scale to 0-1
obj_digit_array = X[obj_idx].reshape(28, 28) / 255.0  # Scale to 0-1

# Create the joint input plane using the new function
# The display_scale_factor in create_joint_input_plane (default 0.4) controls the size
a0 = create_joint_input_plane(ref_digit_array, obj_digit_array, 
                              (N, N), img_thresh, display_scale_factor=0.20) # Changed from 0.3 to 0.2 for smaller digits

# -------------------------- main script section ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Binary/Analog Joint Transform Correlator simulation")
    parser.add_argument("--binary-input", action="store_true", default=True,
                        help="Binarize the input plane (default: True)")
    parser.add_argument("--analog-input", dest="binary_input", action="store_false",
                        help="Use analog (grayscale) values for the input plane")
    parser.add_argument("--binary-jps", action="store_true", default=True,
                        help="Binarize the joint power spectrum (default: True)")
    parser.add_argument("--analog-jps", dest="binary_jps", action="store_false",
                        help="Use analog (grayscale) values for the joint power spectrum")
    parser.add_argument("--ref-digit", type=int, default=1,
                        help="Reference digit (default: 1 for MNIST dataset)")
    parser.add_argument("--obj-digit", type=int, default=1,
                        help="Object digit (default: 1 for MNIST dataset)")
    parser.add_argument("--scale", type=float, default=0.2,
                        help="Scale factor for digits (default: 0.2)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion"],
                        help="Dataset to use: mnist or fashion (default: mnist)")
    args = parser.parse_args()
    
    # Load digits from specified dataset using data.py
    X, y = get_test_data(dataset=args.dataset)
    
    # Select digits based on command-line arguments
    ref_indices = np.where(y == args.ref_digit)[0]
    if len(ref_indices) == 0:
        print(f"Error: Digit {args.ref_digit} not found in {args.dataset} dataset.")
        sys.exit(1)
    ref_idx = ref_indices[0]  # Use the first occurrence of the digit
    
    # For object digit, either use same as reference (autocorrelation) or different digit
    if args.ref_digit == args.obj_digit:
        obj_idx = ref_idx  # Autocorrelation
    else:
        obj_indices = np.where(y == args.obj_digit)[0]
        if len(obj_indices) == 0:
            print(f"Error: Digit {args.obj_digit} not found in {args.dataset} dataset.")
            sys.exit(1)
        obj_idx = obj_indices[0]  # Use the first occurrence of the digit
    
    # Reshape from flat vectors to 28x28 images
    ref_digit_array = X[ref_idx].reshape(28, 28) / 255.0  # Scale to 0-1
    obj_digit_array = X[obj_idx].reshape(28, 28) / 255.0  # Scale to 0-1
    
    # Get dataset class names for title
    dataset_name, class_names = get_dataset(args.dataset)
    print(f"Using {dataset_name}: {class_names[args.ref_digit]} vs {class_names[args.obj_digit]}")
    
    # Create the joint input plane using the new function with binarization option
    a0 = create_joint_input_plane(ref_digit_array, obj_digit_array, 
                                 (N, N), img_thresh, 
                                 display_scale_factor=args.scale,
                                 binarize=args.binary_input)
    
    # Convert grayscale input (0-1) to phase if not binary
    if not args.binary_input:
        # For analog input, scale from 0-1 to 0-π
        phase_a0 = a0 * np.pi
    else:
        # For binary input, map from -1/1 to 0/π
        phase_a0 = (a0 + 1) / 2 * np.pi
    
    # ------------------- optical simulation with LightPipes ---------------------
    # 1. write input on the SLM as phase
    F1 = Begin(slm_size, wavelength, N)
    F1 = MultPhase(F1, phase_a0)
    
    # Apply SLM fill factor (pixel aperture) - amplitude modulation
    # Create coordinate grid for the SLM
    x = np.linspace(-slm_size/2, slm_size/2, N)
    X, Y = np.meshgrid(x, x)
    
    # Generate the amplitude mask for the SLM pixels
    amp_mask = slm_pixel_aperture(X, Y, slm_size, slm_size)
    F1 = MultIntensity(F1, amp_mask)
    
    # 2. Fourier transform to obtain JPS
    F1 = Lens(F1, f_lens)
    F1 = Forvard(F1, f_lens)
    JPS_int = Intensity(F1, 0)
    
    # 3. Process the JPS (binarize or use as is based on the option)
    if args.binary_jps:
        # Threshold the JPS to +1/-1
        thr_JPS = np.median(JPS_int)
        JPS_processed = np.where(JPS_int > thr_JPS, 1.0, -1.0)
        # Map from -1/1 to 0/π
        phase_JPS = (JPS_processed + 1) / 2 * np.pi
        jps_display = JPS_processed  # For visualization
        jps_title = 'Binary JPS (Fourier plane)'
    else:
        # Normalize JPS to 0-1 for analog operation
        JPS_normalized = JPS_int / JPS_int.max() if JPS_int.max() > 0 else JPS_int
        # For analog JPS, scale from 0-1 to 0-π
        phase_JPS = JPS_normalized * np.pi
        jps_display = JPS_normalized  # For visualization
        jps_title = 'Analog JPS (Fourier plane)'
    
    # 4. write JPS back onto the SLM
    F2 = Begin(slm_size, wavelength, N)
    F2 = MultPhase(F2, phase_JPS)
    
    # Apply the same SLM fill factor (pixel aperture) to the JPS
    F2 = MultIntensity(F2, amp_mask)
    
    # 5. second FT to correlation plane
    F2 = Lens(F2, f_lens)
    F2 = Forvard(F2, f_lens)
    Corr_int = Intensity(F2, 0)
    
    # ---------------------------- visualisation ---------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    
    # Input plane display
    if args.binary_input:
        axs[0].set_title('Binary input plane')
        axs[0].imshow(a0, cmap='gray', vmin=-1, vmax=1)
    else:
        axs[0].set_title('Analog input plane')
        axs[0].imshow(a0, cmap='gray', vmin=0, vmax=1)
    axs[0].axis('off')
    
    # JPS display
    axs[1].set_title(jps_title)
    if args.binary_jps:
        axs[1].imshow(jps_display, cmap='gray', vmin=-1, vmax=1)
    else:
        axs[1].imshow(jps_display, cmap='viridis', vmin=0)
    axs[1].axis('off')
    
    # Correlation output
    axs[2].set_title(f'Correlation output ({"Binary" if args.binary_jps else "Analog"} JPS)')
    axs[2].imshow(Corr_int, cmap='inferno')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --------------------------- optional metrics -------------------------------
    # Uncomment for quick PSR & SNR estimation on the correlation output.
    """
    peak_val = Corr_int.max()
    peak_pos = np.unravel_index(np.argmax(Corr_int), Corr_int.shape)

    mask = np.ones_like(Corr_int, bool)
    py, px = peak_pos
    mask[max(py-2,0):py+3, max(px-2,0):px+3] = False  # suppress 5×5 around peak

    side_vals = Corr_int[mask]
    psr = peak_val / side_vals.max()
    noise_rms = np.sqrt(np.mean(side_vals**2))
    snr  = peak_val / noise_rms
    print(f"PSR = {psr:.2f}\nSNR = {snr:.2f}")
    """

# End of file
