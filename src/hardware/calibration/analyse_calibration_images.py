"""
This script analyzes camera images of SLM calibration patterns to determine phase modulation characteristics.

It reads camera-captured images from a 'camera_captures/slm_calibration_YYYY-MM-DD' directory,
where the images should be named "cam_mirror000.png", "cam_mirror050.png", etc.,
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
    """Cosine function for fitting the grating pattern"""
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

def extract_mirror_value(filename):
    """Extract mirror value from filename using regex"""
    match = re.search(r"mirror(\d+)", filename)
    if match:
        return int(match.group(1))
    return None

def analyze_calibration_image(image_path):
    """
    Analyze a calibration image to determine the phase shift of the grating.
    
    Args:
        image_path: Path to the calibration image
        
    Returns:
        phase: Phase shift of the fitted cosine function
    """
    # Read image
    img = plt.imread(image_path)
    
    # Convert to grayscale if image has color channels
    if len(img.shape) > 2:
        img = img[:,:,0]  # Use first channel if RGB
    
    # Get middle row of the image
    height = img.shape[0]
    middle_row = img[height // 2, :]
    
    # We're only interested in the right half (grating pattern)
    width = len(middle_row)
    grating_data = middle_row[width // 2:]
    
    # Generate x values (pixel positions)
    x_values = np.arange(len(grating_data))
    
    # Initial guess for cosine parameters
    p0 = [
        (np.max(grating_data) - np.min(grating_data)) / 2,  # amplitude
        1 / 8,  # frequency (assuming period ~8 pixels)
        0,  # phase
        np.mean(grating_data)  # offset
    ]
    
    # Fit cosine function to the data
    try:
        params, _ = curve_fit(cosine_function, x_values, grating_data, p0=p0)
        amplitude, frequency, phase, offset = params
        
        # Normalize phase to be between 0 and 2π
        phase = phase % (2 * np.pi)
        
        # Plot the fit for debugging (optional)
        """
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, grating_data, 'b-', label='Grating data')
        plt.plot(x_values, cosine_function(x_values, *params), 'r-', label='Fit')
        plt.title(f"Cosine fit for {Path(image_path).name} - Phase: {phase:.4f}")
        plt.legend()
        plt.savefig(f"fit_{Path(image_path).stem}.png")
        plt.close()
        """
        
        return phase
    
    except RuntimeError:
        print(f"Failed to fit {image_path}. Using default phase 0.")
        return 0.0

def main():
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
            
        phase = analyze_calibration_image(file_path)
        
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
    main()


