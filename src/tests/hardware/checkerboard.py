"""
Generate a 1920×1200 checkerboard pattern with alternating grayscale values.
The pattern alternates between black (0) and gray (160) pixels.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_checkerboard(width, height, black_value=0, white_value=160):
    """
    Create a checkerboard pattern of specified dimensions.
    
    Parameters:
    -----------
    width, height : int
        Dimensions of the output image
    black_value : int
        Grayscale value for "black" squares (0-255)
    white_value : int
        Grayscale value for "white" squares (0-255)
    
    Returns:
    --------
    numpy.ndarray
        The checkerboard pattern array
    """
    # Create an array of zeros with the specified shape
    checkerboard = np.zeros((height, width), dtype=np.uint8)
    
    # Set alternating pixels to white_value
    # This creates a 1x1 pixel checkerboard pattern
    checkerboard[0::2, 0::2] = white_value  # Even rows, even columns
    checkerboard[1::2, 1::2] = white_value  # Odd rows, odd columns
    
    return checkerboard

def main():
    # Set dimensions
    width = 1920
    height = 1200
    
    # Create the checkerboard
    print(f"Creating {width}×{height} checkerboard pattern...")
    cb = create_checkerboard(width, height, black_value=0, white_value=160)
    
    # Save the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, f"checkerboard_{width}x{height}.png")
    
    # Save as PNG using PIL
    Image.fromarray(cb).save(output_path)
    print(f"Saved checkerboard to: {output_path}")
    
    # Optional: Display the pattern (showing only a small section for visibility)
    plt.figure(figsize=(10, 6))
    
    # Show the full image (downsampled)
    plt.subplot(121)
    plt.title(f"Full Checkerboard ({width}×{height})")
    plt.imshow(cb, cmap='gray', vmin=0, vmax=255)
    
    # Show a small section to see the pixel alternation
    plt.subplot(122)
    plt.title("Zoomed Section (10×10 pixels)")
    plt.imshow(cb[:10, :10], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "checkerboard_preview.png"))
    plt.show()

if __name__ == "__main__":
    main()