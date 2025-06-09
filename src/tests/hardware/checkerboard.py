"""
Generate a 1920×1200 checkerboard pattern with alternating grayscale values.
The pattern alternates between black (0) and gray (160) pixels.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def create_checkerboard(width, height, block_size=5, black_value=0, white_value=160):
    """
    Create a checkerboard pattern of specified dimensions.
    
    Parameters:
    -----------
    width, height : int
        Dimensions of the output image
    block_size : int
        Size of each checker square in pixels
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
    
    # Calculate how many complete blocks we can fit
    h_blocks = height // block_size
    w_blocks = width // block_size
    
    # Create a binary pattern for the blocks (0 or 1)
    block_pattern = np.zeros((h_blocks, w_blocks), dtype=np.uint8)
    block_pattern[0::2, 0::2] = 1  # Even rows, even columns
    block_pattern[1::2, 1::2] = 1  # Odd rows, odd columns
    
    # Expand the pattern to the full image size
    for i in range(h_blocks):
        for j in range(w_blocks):
            y_start = i * block_size
            y_end = (i + 1) * block_size
            x_start = j * block_size
            x_end = (j + 1) * block_size
            
            # Fill the block with the appropriate value
            value = white_value if block_pattern[i, j] == 1 else black_value
            checkerboard[y_start:y_end, x_start:x_end] = value
    
    # Handle any remaining pixels at the edges if dimensions aren't multiples of block_size
    if height % block_size != 0 or width % block_size != 0:
        for i in range(h_blocks):
            for j in range(w_blocks, width):
                value = white_value if block_pattern[i, w_blocks-1] == 1 else black_value
                checkerboard[i*block_size:min((i+1)*block_size, height), j] = value
        
        for i in range(h_blocks, height):
            for j in range(width):
                if j < w_blocks * block_size:
                    col_block = j // block_size
                    value = white_value if block_pattern[h_blocks-1, col_block] == 1 else black_value
                else:
                    value = white_value if block_pattern[h_blocks-1, w_blocks-1] == 1 else black_value
                checkerboard[i, j] = value
    
    return checkerboard

def main():
    # Set dimensions
    width = 1920
    height = 1200
    block_size = 5  # 5x5 pixel blocks
    
    # Create the checkerboard
    print(f"Creating {width}×{height} checkerboard pattern with {block_size}×{block_size} pixel blocks...")
    cb = create_checkerboard(width, height, block_size=block_size, black_value=0, white_value=160)
    
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
    plt.show()

if __name__ == "__main__":
    main()