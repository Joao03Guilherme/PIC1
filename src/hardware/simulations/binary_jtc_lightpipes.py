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

EPS = 1e-6  # Small constant to prevent division by zero in correlation metrics

# -------------------------- user‑adjustable parameters ----------------------
wavelength = 633e-9               # 633 nm He‑Ne laser
slm_size   = 6e-3                 # 6 mm × 6 mm SLM aperture
N          = 512                  # simulation grid (N × N)
f_lens     = 100e-3               # 100 mm focal‑length Fourier lens
pixel_pitch = slm_size / N        # SLM pixel pitch (physical size of each pixel)

# binary threshold for images (0–255); set to None to use the pixel median
img_thresh = None

# ---------------------------------------------------------------------------
# Correlation function to eliminate code duplication
# ---------------------------------------------------------------------------
def perform_jtc_correlation(a0_pair, binary_input=True, binary_jps=True, blocking_factor=0.1, digit1=None, digit2=None):
    """
    Perform JTC correlation using LightPipes optical simulation.
    
    Parameters:
    -----------
    a0_pair : np.ndarray
        Joint input plane containing the two images side-by-side
    binary_input : bool
        Whether the input plane is binary (True) or grayscale (False)
    binary_jps : bool
        Whether to binarize the joint power spectrum (True) or use as is (False)
    blocking_factor : float
        Factor to determine the size of central region to block (as portion of N)
    digit1 : int or None
        First digit label (optional, used for directional search)
    digit2 : int or None
        Second digit label (optional, used for directional search)
        
    Returns:
    --------
    tuple
        (peak_val, central_peak_intensity, peak_coords, Corr_int)
        - peak_val: Value of the highest correlation peak after blocking central region
        - central_peak_intensity: Value of the central DC peak
        - peak_coords: (dy, dx) coordinates of the highest peak relative to center
        - Corr_int: Full correlation intensity plane
    """
    # Convert to phase based on whether input is binary or analog
    if binary_input:
        phase_pair = (a0_pair + 1) / 2 * np.pi  # Map from -1/1 to 0/π
    else:
        phase_pair = a0_pair * np.pi  # Scale from 0-1 to 0-π
    
    # First optical pass (compute JPS)
    F1 = Begin(slm_size, wavelength, N)
    F1 = MultPhase(F1, phase_pair)
    F1 = Lens(F1, f_lens)
    F1 = Forvard(F1, f_lens)
    JPS_int = Intensity(F1, 0)
    
    # Process JPS (binarize if needed)
    if binary_jps:
        thr_JPS = np.median(JPS_int)
        JPS_processed = np.where(JPS_int > thr_JPS, 1.0, -1.0)
        phase_JPS = (JPS_processed + 1) / 2 * np.pi  # Map from -1/1 to 0/π
    else:
        JPS_normalized = JPS_int / JPS_int.max() if JPS_int.max() > 0 else JPS_int
        phase_JPS = JPS_normalized * np.pi  # Scale from 0-1 to 0-π
    
    # Second optical pass (correlation)
    F2 = Begin(slm_size, wavelength, N)
    F2 = MultPhase(F2, phase_JPS)
    F2 = Lens(F2, f_lens)
    F2 = Forvard(F2, f_lens)
    Corr_int = Intensity(F2, 0)
    
    # Get central peak intensity (DC term) before masking
    center_y, center_x = N // 2, N // 2
    central_peak_intensity = Corr_int.max()
    
    # Block central region to find correlation peak
    corr_masked = Corr_int.copy()
    dc_block_half_width = int(N * blocking_factor)
    ystart = max(0, center_y - dc_block_half_width)
    yend = min(N, center_y + dc_block_half_width + 1)
    xstart = max(0, center_x - dc_block_half_width)
    xend = min(N, center_x + dc_block_half_width + 1)
    corr_masked[ystart:yend, xstart:xend] = 0.0
    
    # Instead of using digit values, we'll examine the correlation pattern
    # and find peaks on both sides, then decide which one to use
    
    # First, try to find peaks in left and right halves separately
    left_half = corr_masked.copy()
    left_half[:, center_x:] = 0.0  # Keep only left half
    
    right_half = corr_masked.copy()
    right_half[:, :center_x] = 0.0  # Keep only right half
    
    # Find max peaks in each half
    left_peak_val = left_half.max()
    left_peak_idx = np.unravel_index(np.argmax(left_half), left_half.shape)
    left_peak_dy = left_peak_idx[0] - center_y
    left_peak_dx = left_peak_idx[1] - center_x
    
    right_peak_val = right_half.max()
    right_peak_idx = np.unravel_index(np.argmax(right_half), right_half.shape)
    right_peak_dy = right_peak_idx[0] - center_y
    right_peak_dx = right_peak_idx[1] - center_x
    
    # Choose the half with the stronger peak
    if left_peak_val > right_peak_val:
        # Use left peak
        peak_val = left_peak_val
        dy = left_peak_dy
        dx = left_peak_dx
        search_region = "left half (stronger peak)"
        # Update corr_masked to show only the left half in visualization
        corr_masked = left_half
    else:
        # Use right peak
        peak_val = right_peak_val
        dy = right_peak_dy
        dx = right_peak_dx
        search_region = "right half (stronger peak)"
        # Update corr_masked to show only the right half in visualization
        corr_masked = right_half
    
    # Get peak and calculate coordinates
    peak_val = corr_masked.max()
    peak_idx = np.unravel_index(np.argmax(corr_masked), corr_masked.shape)
    
    # Calculate shift from center
    dy = peak_idx[0] - center_y
    dx = peak_idx[1] - center_x
    
    print(f"Finding correlation peak in {search_region}: peak at ({dx}, {dy})")
    
    return peak_val, central_peak_intensity, (dy, dx), Corr_int, corr_masked

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
                             display_scale_factor: float = 0.2,
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
        

# ---------------------------------------------------------------------
# Compare Euclidean distance vs JTC distance for random MNIST pairs
# ---------------------------------------------------------------------
def compare_euclidean_vs_jtc(dataset="mnist", num_pairs=100, binary_input=True, binary_jps=True, scale=0.2):
    """
    Compare Euclidean distances vs JTC distances using LightPipes simulation.
    
    Parameters:
    -----------
    dataset : str
        Dataset to use ('mnist' or 'fashion')
    num_pairs : int
        Number of random digit pairs to compare
    binary_input : bool
        Whether to binarize the input plane
    binary_jps : bool
        Whether to binarize the joint power spectrum
    scale : float
        Scale factor for digits on the SLM
        
    Returns:
    --------
    tuple
        (euclidean_distances, jtc_distances, r_squared)
    """
    print(f"\nComparing Euclidean vs JTC distances for {num_pairs} random {dataset} pairs...")
    
    # Load dataset
    X_data, y_data = get_test_data(dataset=dataset)
    shape = (28, 28)  # MNIST digit shape
    
    # Lists to store distances
    eucl_dists = []
    jtc_dists = []
    pairs_info = []
    
    # Store the actual images of each pair
    digit_pairs = []
    
    # Process random digit pairs
    for i in range(num_pairs):
        if i % 10 == 0:
            print(f"Processing pair {i}/{num_pairs}...")
        
        # Randomly select two digit samples
        idx1 = np.random.randint(0, len(X_data))
        idx2 = np.random.randint(0, len(X_data))
        
        # Get the digit images and labels
        img1 = X_data[idx1].reshape(shape).astype(np.float32)
        img2 = X_data[idx2].reshape(shape).astype(np.float32)
        lbl1 = y_data[idx1]
        lbl2 = y_data[idx2]
        
        # Store the original images
        digit_pairs.append((img1, img2))
        
        # Calculate Euclidean distance
        eucl_dist = np.linalg.norm(img1.flatten() - img2.flatten())
        eucl_dists.append(eucl_dist)
        
        # Prepare joint input plane
        a0_pair = create_joint_input_plane(img1, img2, (N, N), img_thresh, 
                                           display_scale_factor=scale,
                                           binarize=binary_input)
        
        # Use the reusable correlation function with digit information
        peak_val, central_peak_intensity, (dy, dx), Corr_int, corr_masked = perform_jtc_correlation(
            a0_pair, binary_input=binary_input, binary_jps=binary_jps, blocking_factor=0.01,
            digit1=lbl1, digit2=lbl2  # Pass digit information
        )
        
        # Normalize peak by central peak intensity and convert to distance
        # Using central peak intensity for normalization instead of product of norms
        similarity = peak_val / central_peak_intensity
        jtc_dist = 1.0 / (similarity + EPS)
        
        jtc_dists.append(jtc_dist)
        pairs_info.append((lbl1, lbl2, similarity, (dy, dx)))
    
    # Convert to numpy arrays
    eucl_dists = np.array(eucl_dists)
    jtc_dists = np.array(jtc_dists)
    
    # Fit a line to the data
    m, c = np.polyfit(jtc_dists, eucl_dists, 1)
    y_pred = m * jtc_dists + c
    
    # Calculate R^2
    ss_res = np.sum((eucl_dists - y_pred) ** 2)
    ss_tot = np.sum((eucl_dists - eucl_dists.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + EPS)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculate distance of each point from the regression line
    # The distance from point (x,y) to line ax + by + c = 0 is |ax + by + c|/√(a² + b²)
    # Our line is y = mx + c, which can be rewritten as mx - y + c = 0 (a=m, b=-1, c=c)
    point_distances = np.abs(m * jtc_dists - eucl_dists + c) / np.sqrt(m**2 + 1)
    
    # Find indices of the top 5 outliers (points furthest from the regression line)
    num_outliers = 5
    outlier_indices = np.argsort(point_distances)[-num_outliers:]
    
    # Get the outlier pairs info
    outlier_pairs = [pairs_info[i] for i in outlier_indices]
    
    # Plot data points (non-outliers)
    mask = np.ones(len(jtc_dists), dtype=bool)
    mask[outlier_indices] = False
    ax.scatter(jtc_dists[mask], eucl_dists[mask], alpha=0.7, c='blue', marker='o', 
              s=30, edgecolor='k', linewidth=0.5)
    
    # Plot outliers with different color/style
    ax.scatter(jtc_dists[outlier_indices], eucl_dists[outlier_indices], alpha=1.0, 
               c='red', marker='*', s=120, edgecolor='k', linewidth=1.0, 
               label=f'Top {num_outliers} outliers')
    
    # Add labels for outlier points
    for i, idx in enumerate(outlier_indices):
        pair = pairs_info[idx]
        label = f"{pair[0]} vs {pair[1]}"  # Label with digit pair (e.g., "1 vs 7")
        ax.annotate(label, 
                   (jtc_dists[idx], eucl_dists[idx]),
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    # Plot the best fit line
    xfit = np.linspace(jtc_dists.min(), jtc_dists.max(), 100)
    yfit = m * xfit + c
    ax.plot(xfit, yfit, 'r--', linewidth=2, 
           label=f'Fit: y = {m:.2f}x + {c:.2f}\n$R^2$ = {r2:.3f}')
    
    # Formatting the plot
    ax.set_xlabel('JTC Distance (LightPipes Simulation)', fontsize=12)
    ax.set_ylabel('Euclidean Distance', fontsize=12)
    ax.set_title(f'Euclidean vs JTC Distance ({num_pairs} random {dataset} pairs)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
        
    plt.tight_layout()
    plt.savefig(f'euclidean_vs_jtc_{dataset}_{num_pairs}pairs_with_outliers.png')
    
    # Find the biggest outlier (furthest from regression line)
    biggest_outlier_idx = outlier_indices[-1]  # Last one has highest distance
    biggest_outlier_info = pairs_info[biggest_outlier_idx]
    biggest_outlier_digits = digit_pairs[biggest_outlier_idx]
    
    # Plot the digit pair for the biggest outlier
    plt.figure(figsize=(9, 4))
    
    # Plot the two digits side by side
    plt.subplot(1, 2, 1)
    plt.imshow(biggest_outlier_digits[0], cmap='gray', vmin=0, vmax=255)
    plt.title(f"First digit: {biggest_outlier_info[0]}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(biggest_outlier_digits[1], cmap='gray', vmin=0, vmax=255)
    plt.title(f"Second digit: {biggest_outlier_info[1]}")
    plt.axis('off')
    
    # Add overall title
    plt.suptitle(
        f"Biggest outlier: {biggest_outlier_info[0]} vs {biggest_outlier_info[1]}\n"
        f"JTC Distance: {jtc_dists[biggest_outlier_idx]:.2f}, "
        f"Euclidean Distance: {eucl_dists[biggest_outlier_idx]:.2f}\n"
        f"Distance from regression line: {point_distances[biggest_outlier_idx]:.2f}",
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig(f'biggest_outlier_{dataset}_{num_pairs}pairs.png')
    plt.show()
    
    print(f"Analysis complete. Correlation coefficient (R^2): {r2:.3f}")
    print(f"Biggest outlier: Digits {biggest_outlier_info[0]} vs {biggest_outlier_info[1]}")
    return eucl_dists, jtc_dists, r2

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
    # Use our reusable correlation function to perform JTC
    peak_val, central_peak_intensity, (dy, dx), Corr_int, corr_masked = perform_jtc_correlation(
        a0, binary_input=args.binary_input, binary_jps=args.binary_jps, blocking_factor=0.05,
        digit1=args.ref_digit, digit2=args.obj_digit  # Pass digit information
    )
    
    # Get JPS for display (need to recompute this part for visualization only)
    # Convert to phase based on whether input is binary or analog
    if args.binary_input:
        phase_pair = (a0 + 1) / 2 * np.pi  # Map from -1/1 to 0/π
    else:
        phase_pair = a0 * np.pi  # Scale from 0-1 to 0-π
    
    # First optical pass (compute JPS)
    F1 = Begin(slm_size, wavelength, N)
    F1 = MultPhase(F1, phase_pair)
    F1 = Lens(F1, f_lens)
    F1 = Forvard(F1, f_lens)
    JPS_int = Intensity(F1, 0)
    
    # Process JPS for display purposes
    if args.binary_jps:
        thr_JPS = np.median(JPS_int)
        jps_display = np.where(JPS_int > thr_JPS, 1.0, -1.0)
        jps_title = 'Binary JPS (Fourier plane)'
    else:
        jps_display = JPS_int / JPS_int.max() if JPS_int.max() > 0 else JPS_int
        jps_title = 'Analog JPS (Fourier plane)'
    
    # ---------------------------- visualisation ---------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    
    # Input plane display
    if args.binary_input:
        axs[0, 0].set_title('Binary input plane')
        axs[0, 0].imshow(a0, cmap='gray', vmin=-1, vmax=1)
    else:
        axs[0, 0].set_title('Analog input plane')
        axs[0, 0].imshow(a0, cmap='gray', vmin=0, vmax=1)
    axs[0, 0].axis('off')
    
    # JPS display
    axs[0, 1].set_title(jps_title)
    if args.binary_jps:
        axs[0, 1].imshow(jps_display, cmap='gray', vmin=-1, vmax=1)
    else:
        axs[0, 1].imshow(jps_display, cmap='viridis', vmin=0)
    axs[0, 1].axis('off')
    
    # Correlation output (full)
    axs[1, 0].set_title(f'Full Correlation output ({"Binary" if args.binary_jps else "Analog"} JPS)')
    axs[1, 0].imshow(Corr_int, cmap='inferno')
    axs[1, 0].axis('off')
    
    # Masked Correlation output (shows which side we're searching)
    axs[1, 1].set_title(f'Masked Correlation (Peak search region for {args.ref_digit} vs {args.obj_digit})')
    axs[1, 1].imshow(corr_masked, cmap='inferno')
    # Mark the peak location
    center_y, center_x = N // 2, N // 2
    peak_y = center_y + dy
    peak_x = center_x + dx
    axs[1, 1].plot(peak_x, peak_y, 'wo', markersize=8, markeredgecolor='black', markeredgewidth=1.5)
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    # plt.show() # Will be called once at the end

    # --------------------------- optional metrics -------------------------------
    # Calculate and print specific correlation metrics
    
    # peak_val and central_peak_intensity are returned by perform_jtc_correlation
    
    # EPS is defined globally at the top of the script
    if central_peak_intensity < EPS: # Avoid division by zero if plane is dark
        print("\n--- Correlation Metrics ---")
        print("Warning: Central peak intensity is near zero. Correlation metrics might be unreliable.")
        print(f"Central Peak Intensity (DC term): {central_peak_intensity:.4e}")
        normalized_correlation_value = 0.0
        print(f"Normalized Correlation Value: {normalized_correlation_value:.4f} (due to near-zero central peak)")
        print(f"---------------------------\n")
    else:
        # Calculate the normalized correlation value using central peak intensity
        # This matches our new normalization method in compare_euclidean_vs_jtc
        normalized_correlation_value = peak_val / central_peak_intensity
        
        print(f"\n--- Correlation Metrics ---")
        print(f"Central Peak Intensity (DC term): {central_peak_intensity:.4e}")
        print(f"Max Off-Axis Peak Intensity (after DC block): {peak_val:.4e}")
        print(f"Normalized Correlation Value (Off-Axis Peak / Central Peak): {normalized_correlation_value:.4f}")
        print(f"---------------------------\n")

        compare_euclidean_vs_jtc(num_pairs=1000)
    
    plt.show() # Show all figures





