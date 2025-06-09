"""
Binarize an SLM image using nearest neighbor average thresholding.

This script:
1. Loads the image "SLM.png" from the current directory
2. Binarizes it using the average of each pixel's 4 nearest neighbors as threshold
3. Maps binary values (1, -1) to grayscale values (160, 0)
4. Saves the result as "SLM_binarized.png"
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def nearest_neighbor_average_binarization(image):
    """
    Binarize an image where each pixel is compared to the average
    of its 4 nearest neighbors (above, below, left, right).
    
    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image
        
    Returns:
    --------
    np.ndarray
        Binary image with values 1.0 (if pixel >= neighbor avg) or -1.0
    """
    # Create output array of same shape
    height, width = image.shape
    output = np.zeros_like(image, dtype=float)
    
    # Process each pixel (except border)
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Get the 4 nearest neighbors
            neighbors = [
                image[y-1, x],  # above
                image[y+1, x],  # below
                image[y, x-1],  # left
                image[y, x+1]   # right
            ]
            
            # Calculate average of neighbors
            neighbor_avg = sum(neighbors) / 4.0
            
            # Apply threshold rule
            if image[y, x] >= neighbor_avg:
                output[y, x] = 1.0
            else:
                output[y, x] = -1.0
    
    # Handle border pixels (use global average as threshold for simplicity)
    global_avg = np.mean(image)
    
    # Top and bottom rows
    for x in range(width):
        output[0, x] = 1.0 if image[0, x] >= global_avg else -1.0
        output[height-1, x] = 1.0 if image[height-1, x] >= global_avg else -1.0
    
    # Left and right columns (excluding corners that were already handled)
    for y in range(1, height-1):
        output[y, 0] = 1.0 if image[y, 0] >= global_avg else -1.0
        output[y, width-1] = 1.0 if image[y, width-1] >= global_avg else -1.0
    
    return output

def map_binary_to_grayscale(binary_image, high_value=160, low_value=0):
    """
    Map binary values to grayscale values.
    
    Parameters:
    -----------
    binary_image : np.ndarray
        Binary image with values 1.0 or -1.0
    high_value : int
        Grayscale value to map to 1.0 (default: 160)
    low_value : int
        Grayscale value to map to -1.0 (default: 0)
        
    Returns:
    --------
    np.ndarray
        Grayscale image with uint8 data type
    """
    # Create a copy to avoid modifying the original
    grayscale = np.zeros_like(binary_image, dtype=np.uint8)
    
    # Map values
    grayscale[binary_image > 0] = high_value
    grayscale[binary_image <= 0] = low_value
    
    return grayscale

def main():
    # Path to the input image
    input_path = "SLM.png"
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find '{input_path}' in the current directory.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please make sure the image file is in the correct location.")
        return
    
    # Load the image
    print(f"Loading image: {input_path}")
    try:
        img = np.array(Image.open(input_path).convert('L'))
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print(f"Image loaded successfully. Shape: {img.shape}")
    
    # Step 1: Binarize using nearest neighbor average comparison
    print("Applying nearest neighbor average binarization...")
    binary_img = nearest_neighbor_average_binarization(img)
    
    # Step 2: Map binary values to specified grayscale values
    print("Mapping binary values to grayscale (160 or 0)...")
    grayscale_img = map_binary_to_grayscale(binary_img, high_value=160, low_value=0)
    
    # Save the result
    output_path = "SLM_binarized.png"
    Image.fromarray(grayscale_img).save(output_path)
    print(f"Binarized image saved to: {output_path}")
    
    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title(f"Original Image")
    axs[0].axis('off')
    
    # Binarized image
    axs[1].imshow(grayscale_img, cmap='gray')
    axs[1].set_title(f"Binarized (160/0)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("binarization_comparison.png")
    plt.show()
    print("Comparison visualization saved to: binarization_comparison.png")

if __name__ == "__main__":
    main()