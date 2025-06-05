slm_dimensions = (1920, 1080)  # SLM resolution (width, height)
mirror_pixel_values = [0, 50, 100, 150, 200, 255]  # Mirror pixel values
grating_step_height = 128 # Grating grayscale value (0 - 128)
grating_step_period = 4 # Grating period in pixels

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def generate_calibration_pattern(mirror_value, grating_height, grating_period, slm_dims):
    """
    Generate a calibration pattern with mirror on left half and grating on right half.
    
    Args:
        mirror_value: Constant grayscale value for the mirror part (0-255)
        grating_height: Maximum grayscale value for the grating (0-255)
        grating_period: Period of the grating in pixels
        slm_dims: Dimensions of the SLM (width, height)
        
    Returns:
        2D numpy array with the calibration pattern
    """
    width, height = slm_dims
    # Create empty image
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Fill left half with constant mirror value
    image[:, :width//2] = mirror_value
    
    # Generate grating pattern for right half
    x = np.arange(width//2)
    grating = np.zeros((height, width//2), dtype=np.uint8)
    
    # Create square wave grating
    for i in range(height):
        grating[i, :] = ((x // (grating_period//2)) % 2) * grating_height
    
    # Fill right half with grating pattern
    image[:, width//2:] = grating
    
    return image

def save_calibration_patterns(output_dir="calibration_patterns"):
    """Generate and save all calibration patterns for given mirror values"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate and save pattern for each mirror value
    for mirror_value in mirror_pixel_values:
        # Generate pattern
        pattern = generate_calibration_pattern(
            mirror_value, 
            grating_step_height, 
            grating_step_period, 
            slm_dimensions
        )
        
        # Save pattern as image file
        filename = f"calibration_mirror{mirror_value:03d}_grating{grating_step_height:03d}_period{grating_step_period}.png"
        filepath = output_path / filename
        plt.imsave(filepath, pattern, cmap='gray', vmin=0, vmax=255)
        print(f"Saved pattern to {filepath}")
        
        # Also display the pattern
        plt.figure(figsize=(10, 6))
        plt.imshow(pattern, cmap='gray', vmin=0, vmax=255)
        plt.colorbar(label='Pixel Value')
        plt.title(f"SLM Calibration Pattern - Mirror: {mirror_value}, Grating: {grating_step_height}, Period: {grating_step_period}")
        plt.tight_layout()
        plt.savefig(output_path / f"preview_mirror{mirror_value:03d}.png")
        plt.close()

if __name__ == "__main__":
    # Generate and save all calibration patterns
    save_calibration_patterns()
    
    # Display example of the first pattern
    example_pattern = generate_calibration_pattern(
        mirror_pixel_values[0], 
        grating_step_height, 
        grating_step_period, 
        slm_dimensions
    )
    
    plt.figure(figsize=(10, 6))
    plt.imshow(example_pattern, cmap='gray', vmin=0, vmax=255)
    plt.colorbar(label='Pixel Value')
    plt.title(f"Example SLM Calibration Pattern - Mirror: {mirror_pixel_values[0]}, Grating Height: {grating_step_height}")
    plt.tight_layout()
    plt.show()
    
    print(f"Generated {len(mirror_pixel_values)} calibration patterns with dimensions {slm_dimensions}.")
    print(f"Mirror values: {mirror_pixel_values}")
    print(f"Grating height: {grating_step_height}, Period: {grating_step_period} pixels")