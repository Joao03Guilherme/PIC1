"""
Software twin of OpticalJTCorrelator (phase-only SLM)

Pass 1: [digit1 | digit2] → SLM-1 (phase) + pixel aperture (amplitude)
        → Lens-1 → record spectrum I1 (camera)

Pass 2: I1 → SLM-2 (phase) + pixel aperture (amplitude)
        → Lens-2 → record correlation I2 (camera)

All propagation via LightPipes Fresnel (no FFTs).
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------
from LightPipes import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from matplotlib import cm  # For colormap in 3D plots
from ...data.data import get_test_data   # MNIST loader

# ------------------------------------------------------------------
# Optical & numerical parameters
# ------------------------------------------------------------------
wavelength  = 633 * nm      # He–Ne
slm_cols    = 1920
slm_rows    = 1080
pixel_pitch = 8 * um        # 8 µm
slm_w       = slm_cols * pixel_pitch
slm_h       = slm_rows * pixel_pitch

window_size = 1.2 * slm_w   # simulation window > SLM size
grid_size   = 2048          # must cover window_size
beam_waist  = 1.1 * slm_w   # expanded Gaussian
f1          = 200 * mm      # focal lengths
f2          = 200 * mm

# ------------------------------------------------------------------
# SLM fill-factor amplitude mask
# ------------------------------------------------------------------
def slm_pixel_aperture(x, y, width, height, duty=0.93):
    """1 on each pixel face, 0 in the gaps (duty<1)."""
    gx = (np.mod(x + width/2,  pixel_pitch) < duty*pixel_pitch)
    gy = (np.mod(y + height/2, pixel_pitch) < duty*pixel_pitch)
    return (gx & gy).astype(float)

# ------------------------------------------------------------------
# MNIST helpers
# ------------------------------------------------------------------
def load_digit_bitmap(d):
    X, y = get_test_data()
    idx  = np.where(y == d)[0]
    if not idx.size:
        raise ValueError(f"digit {d} not found")
    return (X[idx[0]].reshape(28,28) / 255.0)  # in [0,1]

def two_digits_phase_mask(d1, d2, display_scale=0.2):
    """
    Phase mask: two digits (d1, d2) are horizontally stacked,
    then this combined image is scaled and centered on the SLM.
    'display_scale' controls the size of the combined image relative
    to the SLM dimensions (aspect ratio preserved).
    display_scale = 1.0: combined image fits SLM width or height.
    display_scale = 0.5: combined image is 50% of that fit-size, centered.
    The area outside the displayed image has zero phase.
    """
    bmpL, bmpR = load_digit_bitmap(d1), load_digit_bitmap(d2)
    combined_bmp = np.hstack((bmpL, bmpR))
    H_bmp, W_bmp = combined_bmp.shape # Height and Width of the combined bitmap

    # The inner 'mask' function is a closure; it captures 'combined_bmp', 
    # 'H_bmp', 'W_bmp', and 'display_scale' from this outer scope.
    def mask(x_slm_coords, y_slm_coords, slm_physical_width, slm_physical_height):
        # Aspect ratio of the combined bitmap (e.g., for 28x56, aspect_ratio_bmp = 56/28 = 2.0)
        aspect_ratio_bmp = W_bmp / H_bmp

        # Determine the 'base' dimensions if the combined_bmp were to snugly fit the SLM
        # while preserving its aspect ratio.
        # Case 1: Fit to SLM height, calculate required width
        width_if_slm_height_limited = slm_physical_height * aspect_ratio_bmp
        # Case 2: Fit to SLM width, calculate required height
        height_if_slm_width_limited = slm_physical_width / aspect_ratio_bmp

        if width_if_slm_height_limited <= slm_physical_width:
            # SLM height is the primary constraint for the 'snug fit'
            base_h_on_slm = slm_physical_height
            base_w_on_slm = width_if_slm_height_limited
        else:
            # SLM width is the primary constraint
            base_w_on_slm = slm_physical_width
            base_h_on_slm = height_if_slm_width_limited

        # Apply the user-provided display_scale to these base dimensions
        actual_display_width_on_slm = base_w_on_slm * display_scale
        actual_display_height_on_slm = base_h_on_slm * display_scale

        output_phase_plane = np.zeros_like(x_slm_coords) # Background phase is 0

        # Define the display box for the combined image, centered on the SLM.
        # SLM coordinates (x_slm_coords, y_slm_coords) are assumed to be centered around (0,0).
        x_display_start_slm = -actual_display_width_on_slm / 2.0
        x_display_end_slm   =  actual_display_width_on_slm / 2.0
        y_display_start_slm = -actual_display_height_on_slm / 2.0
        y_display_end_slm   =  actual_display_height_on_slm / 2.0

        # Create a boolean mask for SLM coordinates that fall within the display box
        active_slm_region_mask = (x_slm_coords >= x_display_start_slm) & (x_slm_coords < x_display_end_slm) & \
                                 (y_slm_coords >= y_display_start_slm) & (y_slm_coords < y_display_end_slm)

        if np.any(active_slm_region_mask):
            # Extract the SLM coordinates that are active
            active_x_slm = x_slm_coords[active_slm_region_mask]
            active_y_slm = y_slm_coords[active_slm_region_mask]

            # Normalize these active SLM coordinates to [0,1) range for bitmap lookup.
            # The origin for these normalized coordinates is the bottom-left of the display box.
            norm_x_for_bmp = (active_x_slm - x_display_start_slm) / actual_display_width_on_slm
            norm_y_for_bmp = (active_y_slm - y_display_start_slm) / actual_display_height_on_slm

            # Convert normalized coordinates to indices for the combined_bmp
            # combined_bmp has dimensions H_bmp (rows), W_bmp (columns)
            col_indices_bmp = np.clip((norm_x_for_bmp * (W_bmp - 1)).astype(int), 0, W_bmp - 1)
            # y-axis for images is typically top-down, so (1 - norm_y_for_bmp) maps SLM's bottom-up y to image's top-down row.
            row_indices_bmp = np.clip(((1 - norm_y_for_bmp) * (H_bmp - 1)).astype(int), 0, H_bmp - 1)

            # Assign phase values from combined_bmp to the active region of the output plane
            output_phase_plane[active_slm_region_mask] = combined_bmp[row_indices_bmp, col_indices_bmp]
        
        return 2 * np.pi * output_phase_plane
    return mask

# ------------------------------------------------------------------
# Pass 1 → spectrum
# ------------------------------------------------------------------
def first_pass(digit1, digit2):
    x = np.linspace(-window_size/2, window_size/2, grid_size)
    X, Y = np.meshgrid(x, x)

    F = Begin(window_size, wavelength, grid_size)
    F = GaussBeam(F, beam_waist)
    F = RectAperture(F, slm_w, slm_h)

    # apply input images as phase
    phase_mask = two_digits_phase_mask(digit1, digit2)
    phase1 = phase_mask(X, Y, slm_w, slm_h)
    F = MultPhase(F, phase1)
    
    # Calculate norms of the individual input images
    bmpL, bmpR = load_digit_bitmap(digit1), load_digit_bitmap(digit2)
    img1_norm = np.linalg.norm(bmpL.flatten())
    img2_norm = np.linalg.norm(bmpR.flatten())

    # apply pixel gaps
    amp_pix = slm_pixel_aperture(X, Y, slm_w, slm_h)
    F = MultIntensity(F, amp_pix)

    # Fourier → spectrum (raw intensity)
    F = Lens(F, f1);  F = Fresnel(F, f1)
    I1 = Intensity(0, F)
    return x, I1, img1_norm, img2_norm

# ------------------------------------------------------------------
# Pass 2 → correlation
# ------------------------------------------------------------------
def second_pass(I1_original): # Renamed to I1_original to avoid confusion
    x = np.linspace(-window_size/2, window_size/2, grid_size)
    X, Y = np.meshgrid(x, x)

    F = Begin(window_size, wavelength, grid_size)
    F = GaussBeam(F, beam_waist)
    F = RectAperture(F, slm_w, slm_h)

    # Binarize the Joint Power Spectrum (I1_original)
    # Using median as threshold
    threshold = np.median(I1_original)
    I1_binary = (I1_original > threshold).astype(float) # Convert boolean to float (0.0 or 1.0)

    # display binarized spectrum as phase
    # The phase will be 0 for pixels below threshold, and 2*pi (effectively 0) for pixels above.
    # To make it a binary phase (e.g., 0 and pi), one might use:
    # phase_for_binary_I1 = I1_binary * np.pi 
    # However, the request is to binarize the JPS, then use it for phase modulation as before.
    # So, if I1_binary is 0 or 1, phase is 0 or 2*pi.
    F = MultPhase(F, 2*np.pi * I1_binary)

    # pixel gaps again
    amp_pix = slm_pixel_aperture(X, Y, slm_w, slm_h)
    F = MultIntensity(F, amp_pix)

    # Fourier → correlation (raw intensity)
    F = Lens(F, f2);  F = Fresnel(F, f2)
    I2 = Intensity(0, F)
    return x, I2

# ------------------------------------------------------------------
# Plot helper (linear or log)
# ------------------------------------------------------------------
def plot_plane(x, I, title, zeros, slm_w, slm_h, ax, log=False, block_dc=0.0):
    fx = x / (wavelength * f1)
    fx0 = 1/slm_w
    lim = zeros * fx0 / 1e3
    fx_mm = fx/1e3
    
    # Create a copy of the intensity array to avoid modifying the original
    I_display = I.copy()
    
    # Block DC component if requested
    if block_dc > 0:
        # Calculate the center of the array
        center_x = I_display.shape[1] // 2
        
        # Calculate block size as a fraction of the display width
        block_size_x = int(I_display.shape[1] * block_dc / 2)
        
        # Mask out the central vertical slit (set to zero)
        I_display[:, center_x-block_size_x:center_x+block_size_x] = 0
        
        # Add note to title if blocking is applied
        title = f"{title} (vertical slit DC blocked)"

    if log:
        im = ax.imshow(I_display+1e-12,
                       extent=[fx_mm[0], fx_mm[-1], fx_mm[0], fx_mm[-1]],
                       origin="lower", cmap="viridis",
                       norm=LogNorm(vmin=I_display.max()*1e-6, vmax=I_display.max()),
                       aspect="equal")
        cb_label = "log₁₀ intensity"
    else:
        im = ax.imshow(I_display,
                       extent=[fx_mm[0], fx_mm[-1], fx_mm[0], fx_mm[-1]],
                       origin="lower", cmap="viridis", aspect="equal")
        cb_label = "intensity"

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$f_x\ (\mathrm{cycles/mm})$")
    ax.set_ylabel(r"$f_y\ (\mathrm{cycles/mm})$")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cb_label)

# ------------------------------------------------------------------
# 3D Plot helper
# ------------------------------------------------------------------
def plot_3d_plane(x, I, title, zeros, slm_w, slm_h, ax, log=False, block_dc=0.0):
    """Create a 3D surface plot of the intensity distribution."""
    fx = x / (wavelength * f1)
    fx0 = 1/slm_w
    lim = zeros * fx0 / 1e3
    fx_mm = fx/1e3
    
    # Get indices for the region of interest based on the zoom factor
    idx_min = np.abs(fx_mm - (-lim)).argmin()
    idx_max = np.abs(fx_mm - lim).argmin()
    
    # Extract the region of interest
    roi_x = fx_mm[idx_min:idx_max]
    roi_y = fx_mm[idx_min:idx_max]
    roi_I = I[idx_min:idx_max, idx_min:idx_max].copy()  # Make a copy to avoid modifying original
    
    # Block DC component if requested
    if block_dc > 0:
        # Calculate the center of the ROI array
        center_x = roi_I.shape[1] // 2
        
        # Calculate block size as a fraction of the display width
        block_size_x = int(roi_I.shape[1] * block_dc / 2)
        
        # Mask out the central vertical slit (set to zero)
        roi_I[:, center_x-block_size_x:center_x+block_size_x] = 0
        
        # Add note to title if blocking is applied
        title = f"{title} (vertical slit DC blocked)"
    
    # Create mesh grid for 3D plotting
    X, Y = np.meshgrid(roi_x, roi_y)
    
    # Apply log scale if requested
    if log:
        Z = np.log10(roi_I + 1e-12)  # Add small epsilon to avoid log(0)
        z_label = "log₁₀ intensity"
    else:
        Z = roi_I
        z_label = "intensity"
    
    # Create the 3D surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=False)
    
    ax.set_xlabel(r"$f_x\ (\mathrm{cycles/mm})$")
    ax.set_ylabel(r"$f_y\ (\mathrm{cycles/mm})$")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    
    # Add a color bar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=z_label)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase-only JTC sim with log option")
    p.add_argument("--d1",   type=int,   default=1,   help="left digit")
    p.add_argument("--d2",   type=int,   default=2,   help="right digit")
    p.add_argument("--zoom", type=float, default=5.0, help="±n sinc zeros")
    p.add_argument("--log",  action="store_true",     help="use logarithmic scale")
    p.add_argument("--3d",   dest="plot3d", action="store_true", help="show 3D plot of correlation plane in a separate window")
    p.add_argument("--block-dc", type=float, default=0.0, 
                   help="block central region of correlation plane by this fraction of display area (0.0-1.0)")
    args = p.parse_args()

    # Perform simulation passes
    sim_x_coords, I1, img1_norm, img2_norm = first_pass(args.d1, args.d2)
    _, I2 = second_pass(I1)
    
    # Prepare data for input plane plot
    phase_mask_func = two_digits_phase_mask(args.d1, args.d2)
    # sim_x_coords is already available from first_pass
    X_grid, Y_grid = np.meshgrid(sim_x_coords, sim_x_coords)
    input_phase_image = phase_mask_func(X_grid, Y_grid, slm_w, slm_h)

    # Calculate the normalized correlation value (code remains the same)
    fx = sim_x_coords / (wavelength * f1)
    fx0 = 1/slm_w
    lim = args.zoom * fx0 / 1e3
    fx_mm = sim_x_coords/1e3 # Use sim_x_coords here
    
    idx_min = np.abs(fx_mm - (-lim)).argmin()
    idx_max = np.abs(fx_mm - lim).argmin()
    
    roi_I = I2[idx_min:idx_max, idx_min:idx_max].copy()
    
    if args.block_dc > 0:
        center_x_roi = roi_I.shape[1] // 2 # Renamed to avoid conflict
        block_size_x_roi = int(roi_I.shape[1] * args.block_dc / 2) # Renamed
        roi_I[:, center_x_roi-block_size_x_roi:center_x_roi+block_size_x_roi] = 0
    
    max_corr = np.max(roi_I)
    
    norm_product = img1_norm * img2_norm
    if norm_product > 0:
        norm_corr = max_corr / norm_product
        scaled_corr = norm_corr * 1000
        print(f"\nMaximum correlation value (in zoomed, DC-blocked region): {max_corr:.6f}")
        print(f"Product of image norms: {norm_product:.6f}")
        print(f"Normalized correlation value × 1000: {scaled_corr:.6f}")
    else:
        print("\nWarning: Product of image norms is zero, cannot normalize correlation value")
    
    # Create the 3-panel 2D plot
    fig, (ax_input, ax_jps, ax_corr) = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Input Plane
    extent_mm = [sim_x_coords.min()/mm, sim_x_coords.max()/mm, sim_x_coords.min()/mm, sim_x_coords.max()/mm]
    im_input = ax_input.imshow(input_phase_image, extent=extent_mm, origin="lower", cmap="twilight_shifted")
    
    # SLM Aperture
    rect_slm = plt.Rectangle((-slm_w/(2*mm), -slm_h/(2*mm)), slm_w/mm, slm_h/mm, 
                             color='blue', fill=False, linestyle=':', linewidth=1, label=f'SLM ({slm_w/mm:.1f}x{slm_h/mm:.1f}mm)')
    ax_input.add_patch(rect_slm)

    # Beam Waist
    # beam_waist is the 1/e^2 radius
    circle_beam = plt.Circle((0, 0), beam_waist/mm, 
                             color='red', fill=False, linestyle='--', linewidth=1, label=f'Beam Waist ({beam_waist/mm:.1f}mm rad.)')
    ax_input.add_patch(circle_beam)
    
    ax_input.set_title(f"Input Plane (Digits {args.d1}&{args.d2})")
    ax_input.set_xlabel("x (mm)")
    ax_input.set_ylabel("y (mm)")
    ax_input.set_xlim(extent_mm[0], extent_mm[1]) # Match imshow extent
    ax_input.set_ylim(extent_mm[2], extent_mm[3]) # Match imshow extent
    ax_input.legend(fontsize='small')
    fig.colorbar(im_input, ax=ax_input, label="Phase (radians)", shrink=0.8)

    # Panel 2: Joint Power Spectrum
    plot_plane(sim_x_coords, I1,
            f"Spectrum of digits {args.d1}&{args.d2}",
            zeros=args.zoom, slm_w=slm_w, slm_h=slm_h,
            ax=ax_jps, log=args.log)

    # Panel 3: Correlation Plane
    plot_plane(sim_x_coords, I2,
            "Correlation plane",
            zeros=args.zoom, slm_w=slm_w, slm_h=slm_h,
            ax=ax_corr, log=args.log, block_dc=args.block_dc)
    
    plt.tight_layout()
    plt.show() # Show the 2D plots

    # Conditional 3D plot in a new figure
    if args.plot3d:
        fig3d = plt.figure(figsize=(10, 8))
        ax_3d_corr_plot = fig3d.add_subplot(1, 1, 1, projection='3d') # Renamed to avoid conflict
        
        plot_3d_plane(sim_x_coords, I2,
                    f"3D Correlation plane (digits {args.d1}&{args.d2})",
                    zeros=args.zoom, slm_w=slm_w, slm_h=slm_h,
                    ax=ax_3d_corr_plot, log=args.log, block_dc=args.block_dc)
        plt.tight_layout()
        plt.show() # Show the 3D plot
