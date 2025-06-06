"""
This script analyzes camera images of SLM calibration patterns to determine phase modulation characteristics.

It reads camera-captured images from a 'camera_captures/slm_calibration_YYYY-MM-DD' directory,
where the images should be named "cam_mirror000.png", "cam_mirror        if use_alternative_method:
            phase = analyze_calibration_image_alternative(file_path, debug=debug)
        else:
            phase = analyze_calibration_image(file_path, debug=debug)
        
        mirror_values.append(mirror_value)
        phase_shifts.append(phase)
        
        print(f"Mirror value: {mirror_value}, Phase shift: {phase:.4f} rad")g", etc.,
corresponding to different SLM mirror pixel values (0, 50, 100, 150, 200, 255).

For each image, it:
1. Selects a row of pixels in the middle of the image to create a 1D intensity profile.
2. Fits a cosine function to the grating pattern, determining the phase shift.
3. Plots the phase shift against the mirror pixel value and fits a linear relationship.
4. Saves the plot as "calibration_phase_shift.png" with the linear fit included.
5. Prints each data point to the console with R² goodness-of-fit metric.
6. Saves the fit parameters to "calibration_fit_params.txt" for future reference.

The slope of the linear fit represents the SLM's phase modulation efficiency,
which tells how many grayscale levels are needed for a 2π phase shift.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import glob
import re

def cosine_function(x, amplitude, frequency, phase, offset):
    """Cosine function for reference - not used in the improved approach"""
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

def extract_mirror_value(filename):
    """Extract mirror value from filename using regex"""
    match = re.search(r"mirror(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

def analyze_calibration_image(image_path, debug=False):
    """
    Analyze a calibration image to determine the phase shift of the grating
    using Fourier analysis, which is more robust than direct cosine fitting.
    
    Args:
        image_path: Path to the calibration image
        debug: Whether to generate debug plots
        
    Returns:
        phase: Phase shift of the grating pattern
    """
    # Read image
    img = plt.imread(image_path)
    
    # Convert to grayscale if image has color channels
    if len(img.shape) > 2:
        img = img[:,:,0]  # Use first channel if RGB
    
    # Average multiple rows near the middle for better SNR
    height = img.shape[0]
    num_rows_to_average = max(1, height // 20)  # average ~5% of rows for better signal
    middle_start = height // 2 - num_rows_to_average // 2
    middle_end = middle_start + num_rows_to_average
    averaged_rows = np.mean(img[middle_start:middle_end, :], axis=0)
    
    # We're only interested in the right half (grating pattern)
    width = len(averaged_rows)
    grating_data = averaged_rows[width // 2:]
    
    # Normalize data to range [0, 1] to make phase extraction more consistent
    grating_data = (grating_data - np.min(grating_data)) / (np.max(grating_data) - np.min(grating_data))
    
    # Apply windowing to reduce spectral leakage
    window = np.hanning(len(grating_data))
    windowed_data = (grating_data - np.mean(grating_data)) * window
    
    # Perform FFT to find dominant frequency and phase
    fft_result = np.fft.fft(windowed_data)
    magnitudes = np.abs(fft_result)
    
    # Find dominant frequency (excluding DC component)
    dominant_idx = np.argmax(magnitudes[1:len(magnitudes)//2]) + 1
    dominant_magnitude = magnitudes[dominant_idx]
    
    # Get phase at dominant frequency
    phase = np.angle(fft_result[dominant_idx]) % (2 * np.pi)
    
    # Optional: Refine with interpolation for more precise frequency and phase
    
    if debug:
        # Create debug plot showing the data, spectrum, and phase extraction
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot the intensity profile
        axs[0].plot(grating_data, 'b-', label='Grating intensity profile')
        axs[0].set_title(f"Grating Pattern from {Path(image_path).name}")
        axs[0].set_xlabel("Pixel Position")
        axs[0].set_ylabel("Normalized Intensity")
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot FFT magnitude spectrum
        freq = np.fft.fftfreq(len(grating_data))[:len(grating_data)//2]
        axs[1].plot(freq[1:], magnitudes[1:len(magnitudes)//2], 'r-')
        axs[1].axvline(x=freq[dominant_idx], color='g', linestyle='--', 
                      label=f'Dominant freq: {freq[dominant_idx]:.4f}, Phase: {phase:.4f} rad')
        axs[1].set_title("Frequency Spectrum")
        axs[1].set_xlabel("Frequency")
        axs[1].set_ylabel("Magnitude")
        axs[1].grid(True)
        axs[1].legend()
        
        plt.tight_layout()
        debug_dir = Path("debug_plots")
        debug_dir.mkdir(exist_ok=True)
        plt.savefig(debug_dir / f"debug_{Path(image_path).stem}.png")
        plt.close()
        
    return phase

def analyze_calibration_image_alternative(image_path, debug=False):
    """
    Alternative approach for analyzing calibration images using analytical 
    reconstruction to extract phase more reliably.
    
    This method:
    1. Averages multiple rows for better SNR
    2. Determines grating frequency using FFT
    3. Reconstructs a perfect reference cosine at that frequency
    4. Uses cross-correlation to find the phase shift between reference and data
    
    Args:
        image_path: Path to the calibration image
        debug: Whether to generate debug plots
        
    Returns:
        phase: Phase shift of the grating pattern in radians
    """
    # Read image
    img = plt.imread(image_path)
    
    # Convert to grayscale if image has color channels
    if len(img.shape) > 2:
        img = img[:,:,0]  # Use first channel if RGB
    
    # Average multiple rows near the middle for better SNR
    height = img.shape[0]
    num_rows_to_average = max(1, height // 10)  # average 10% of rows
    middle_start = height // 2 - num_rows_to_average // 2
    middle_end = middle_start + num_rows_to_average
    averaged_rows = np.mean(img[middle_start:middle_end, :], axis=0)
    
    # We're only interested in the right half (grating pattern)
    width = len(averaged_rows)
    grating_data = averaged_rows[width // 2:]
    
    # Remove any DC offset
    grating_data = grating_data - np.mean(grating_data)
    
    # Apply windowing to reduce spectral leakage
    window = np.blackman(len(grating_data))
    windowed_data = grating_data * window
    
    # Perform FFT to find dominant frequency
    fft_result = np.fft.rfft(windowed_data)
    magnitudes = np.abs(fft_result)
    
    # Find dominant frequency (excluding very low frequencies)
    min_freq_idx = max(1, len(magnitudes) // 100)  # Skip very low frequencies
    dominant_idx = np.argmax(magnitudes[min_freq_idx:]) + min_freq_idx
    
    # Calculate the dominant frequency in cycles/sample
    dominant_freq = dominant_idx / len(grating_data)
    
    # Generate reference cosine at exactly the same frequency
    x = np.arange(len(grating_data))
    reference_signal = np.cos(2 * np.pi * dominant_freq * x)
    
    # Use Hilbert transform for analytical phase extraction
    from scipy.signal import hilbert
    
    # Create analytic signals
    analytic_data = hilbert(grating_data)
    analytic_ref = hilbert(reference_signal)
    
    # Extract instantaneous phase
    inst_phase_data = np.unwrap(np.angle(analytic_data))
    inst_phase_ref = np.unwrap(np.angle(analytic_ref))
    
    # Calculate phase difference
    phase_diff = np.median(inst_phase_data - inst_phase_ref)
    
    # Normalize to [0, 2π)
    phase = phase_diff % (2 * np.pi)
    
    if debug:
        # Create debug plot showing the data, reference, and phase extraction
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot the intensity profile
        axs[0].plot(grating_data, 'b-', label='Original grating data')
        axs[0].set_title(f"Grating Pattern from {Path(image_path).name}")
        axs[0].set_xlabel("Pixel Position")
        axs[0].set_ylabel("Intensity")
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot reference and shifted signals
        axs[1].plot(reference_signal, 'g-', label='Reference cosine', alpha=0.5)
        axs[1].plot(np.cos(2 * np.pi * dominant_freq * x + phase), 'r-', 
                   label=f'Shifted cosine (phase={phase:.4f} rad)', alpha=0.5)
        axs[1].plot(grating_data, 'b-', label='Original data', alpha=0.3)
        axs[1].set_title(f"Phase Comparison - Extracted phase: {phase:.4f} rad")
        axs[1].set_xlabel("Pixel Position")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True)
        axs[1].legend()
        
        # Plot FFT magnitude spectrum
        freq = np.fft.rfftfreq(len(grating_data))
        axs[2].plot(freq, magnitudes, 'r-')
        axs[2].axvline(x=freq[dominant_idx], color='g', linestyle='--', 
                      label=f'Dominant freq: {freq[dominant_idx]:.4f}')
        axs[2].set_title("Frequency Spectrum")
        axs[2].set_xlabel("Frequency (cycles/sample)")
        axs[2].set_ylabel("Magnitude")
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        debug_dir = Path("debug_plots_alt")
        debug_dir.mkdir(exist_ok=True)
        plt.savefig(debug_dir / f"debug_alt_{Path(image_path).stem}.png")
        plt.close()
        
    return phase

def main(use_alternative_method=False, debug=True):
    """
    Main function to process calibration images and determine SLM phase response.
    
    Args:
        use_alternative_method: If True, uses the alternative phase extraction approach
        debug: Whether to generate debug plots
    """
    # Find all calibration images
    calibration_dir = Path(f"camera_captures")
    
    if not calibration_dir.exists():
        print(f"Calibration directory not found: {calibration_dir}")
        print(f"Creating directory structure for you to save your camera images")
        calibration_dir.mkdir(parents=True, exist_ok=True)
        print(f"Please save your camera images as cam_mirror000.png, cam_mirror050.png, etc. in {calibration_dir}")
        print(f"Then run this script again.")
        return
        
    calibration_files = sorted(glob.glob(str(calibration_dir / "cam_mirror*.png")))
    
    if not calibration_files:
        print(f"No calibration images found in {calibration_dir}")
        return
        
    print(f"Found {len(calibration_files)} calibration images")
    
    # Analyze each image and collect results
    mirror_values = []
    phase_shifts = []
    
    for file_path in calibration_files:
        mirror_value = extract_mirror_value(file_path)
        if mirror_value is None:
            print(f"Could not extract mirror value from {file_path}, skipping")
            continue
            
        if use_alternative_method:
            phase = analyze_calibration_image_alternative(file_path, debug=debug)
        else:
            phase = analyze_calibration_image(file_path, debug=debug)
        
        mirror_values.append(mirror_value)
        phase_shifts.append(phase)
        
        print(f"Mirror value: {mirror_value}, Phase shift: {phase:.4f} rad")
    
    if not mirror_values:
        print("No valid data collected")
        return
        
    # Convert to numpy arrays for fitting
    mirror_values = np.array(mirror_values)
    phase_shifts = np.array(phase_shifts)
    
    # Fit linear function to phase shifts vs mirror values
    fit_params = np.polyfit(mirror_values, phase_shifts, 1)
    slope, intercept = fit_params
    
    # Calculate R-squared for the fit
    y_fit_data = slope * mirror_values + intercept
    residuals = phase_shifts - y_fit_data
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((phase_shifts - np.mean(phase_shifts))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create fit line for plotting
    x_fit = np.linspace(min(mirror_values), max(mirror_values), 100)
    y_fit = slope * x_fit + intercept
    
    # Create and save the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mirror_values, phase_shifts, color='blue', label='Measured data')
    plt.plot(x_fit, y_fit, 'r-', label=f'Linear fit: y = {slope:.6f}x + {intercept:.6f}, R² = {r_squared:.4f}')
    plt.xlabel('Mirror Pixel Value')
    plt.ylabel('Phase Shift (radians)')
    plt.title('SLM Calibration: Phase Shift vs Mirror Pixel Value')
    plt.grid(True)
    plt.legend()
    plt.savefig("calibration_phase_shift.png", dpi=300)
    print(f"Saved plot to calibration_phase_shift.png")
    
    # Save fit parameters to text file
    with open("calibration_fit_params.txt", "w") as f:
        f.write(f"Linear fit parameters for SLM calibration\n")
        f.write(f"Phase shift (radians) = slope * pixel_value + intercept\n")
        f.write(f"slope = {slope}\n")
        f.write(f"intercept = {intercept}\n")
        f.write(f"R-squared = {r_squared}\n")
        f.write(f"\nThis means each grayscale unit corresponds to {slope} radians of phase shift\n")
        f.write(f"Full 2π phase shift requires approximately {2*np.pi/slope:.1f} grayscale units\n")
        f.write(f"Quality of linear fit: R² = {r_squared:.4f} (1.0 is perfect fit)\n")
    
    print(f"Saved fit parameters to calibration_fit_params.txt")
    print(f"Calibration complete: Phase (rad) = {slope:.6f} × pixel_value + {intercept:.6f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze SLM calibration images and extract phase modulation curve')
    parser.add_argument('--alternative', action='store_true', 
                        help='Use alternative phase extraction method (recommended for noisy data)')
    parser.add_argument('--no-debug', action='store_true', 
                        help='Disable debug plots generation')
    
    args = parser.parse_args()
    
    print(f"Using {'alternative' if args.alternative else 'standard'} phase extraction method")
    print(f"Debug plots {'disabled' if args.no_debug else 'enabled'}")
    
    main(use_alternative_method=args.alternative, debug=not args.no_debug)


