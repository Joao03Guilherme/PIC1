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
def slm_pixel_aperture(x, y, width, height, duty=0.90):
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

def two_digits_phase_mask(d1, d2):
    """Phase mask: left = digit d1, right = digit d2, scaled to [0,2π)."""
    bmpL, bmpR = load_digit_bitmap(d1), load_digit_bitmap(d2)
    H, W = bmpL.shape

    def mask(x, y, width, height, bmpL=bmpL, bmpR=bmpR):
        half = width/2
        u = (x + half) / width      # 0…1 across SLM
        v = (y + height/2) / height
        col = np.clip((u * (W-1)).astype(int), 0, W-1)
        row = np.clip(((1-v) * (H-1)).astype(int), 0, H-1)
        flat = np.zeros_like(x)
        left = x < 0
        flat[left]  = bmpL[row[left],  col[left]]
        flat[~left] = bmpR[row[~left], col[~left]]
        return 2*np.pi * flat   # scale to 0…2π
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
    phase1 = two_digits_phase_mask(digit1, digit2)(X, Y, slm_w, slm_h)
    F = MultPhase(F, phase1)

    # apply pixel gaps
    amp_pix = slm_pixel_aperture(X, Y, slm_w, slm_h)
    F = MultIntensity(F, amp_pix)

    # Fourier → spectrum (raw intensity)
    F = Lens(F, f1);  F = Fresnel(F, f1)
    I1 = Intensity(0, F)
    return x, I1

# ------------------------------------------------------------------
# Pass 2 → correlation
# ------------------------------------------------------------------
def second_pass(I1):
    x = np.linspace(-window_size/2, window_size/2, grid_size)
    X, Y = np.meshgrid(x, x)

    F = Begin(window_size, wavelength, grid_size)
    F = GaussBeam(F, beam_waist)
    F = RectAperture(F, slm_w, slm_h)

    # display spectrum as phase
    F = MultPhase(F, 2*np.pi * I1)

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
def plot_plane(x, I, title, zeros, slm_w, slm_h, ax, log=False):
    fx = x / (wavelength * f1)
    fx0 = 1/slm_w
    lim = zeros * fx0 / 1e3
    fx_mm = fx/1e3

    if log:
        im = ax.imshow(I+1e-12,
                       extent=[fx_mm[0], fx_mm[-1], fx_mm[0], fx_mm[-1]],
                       origin="lower", cmap="gray",
                       norm=LogNorm(vmin=I.max()*1e-6, vmax=I.max()),
                       aspect="equal")
        cb_label = "log₁₀ intensity"
    else:
        im = ax.imshow(I,
                       extent=[fx_mm[0], fx_mm[-1], fx_mm[0], fx_mm[-1]],
                       origin="lower", cmap="gray", aspect="equal")
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
def plot_3d_plane(x, I, title, zeros, slm_w, slm_h, ax, log=False):
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
    roi_I = I[idx_min:idx_max, idx_min:idx_max]
    
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
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
    
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
    p.add_argument("--3d",   dest="plot3d", action="store_true", help="show 3D plot of correlation plane")
    args = p.parse_args()

    x, I1 = first_pass(args.d1, args.d2)
    _, I2 = second_pass(I1)

    if args.plot3d:
        # Create a figure with only the 3D plot
        fig = plt.figure(figsize=(10, 8))
        
        # 3D plot
        ax3 = fig.add_subplot(1, 1, 1, projection='3d')
        
        # Create the 3D plot of the correlation plane
        plot_3d_plane(x, I2,
                    f"3D Correlation plane (digits {args.d1}&{args.d2})",
                    zeros=args.zoom, slm_w=slm_w, slm_h=slm_h,
                    ax=ax3, log=args.log)
    else:
        # Standard 2D plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_plane(x, I1,
                f"Spectrum of digits {args.d1}&{args.d2}",
                zeros=args.zoom, slm_w=slm_w, slm_h=slm_h,
                ax=ax1, log=args.log)
        plot_plane(x, I2,
                "Correlation plane",
                zeros=args.zoom, slm_w=slm_w, slm_h=slm_h,
                ax=ax2, log=args.log)
    
    plt.tight_layout()
    plt.show()
