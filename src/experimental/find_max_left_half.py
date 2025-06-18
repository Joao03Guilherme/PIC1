#!/usr/bin/env python3
"""
Find maximum value in the left half of various PNG images.

This script:
1. Searches for all files matching the pattern "[number]v[number]_scale[number].png"
2. Loads each image as an 8-bit integer array
3. Divides the image in half (left and right)
4. Calculates and prints the maximum value in the left half
5. Creates plots of max values for '1vY' and 'Xv1' patterns
"""

import os
import numpy as np
from PIL import Image
import re
import matplotlib.pyplot as plt  # Added import for plotting

def process_image(file_path):
    """
    Process a single image file.
    
    Parameters:
    -----------
    file_path : str
        Path to the image file
        
    Returns:
    --------
    tuple
        (file_name, max_value_left_half, avg_top_values)
    """
    # Extract the file name without path
    file_name = os.path.basename(file_path)
    
    try:
        # Load the image as grayscale
        img = np.array(Image.open(file_path).convert('L'), dtype=np.int32)
        
        # Get image dimensions
        height, width = img.shape
        
        # Divide image in half (left half only)
        left_half = img[:, 0:width//2] 
        
        # Find maximum value in left half (for reference)
        max_value = np.max(left_half)
        
        # Apply Gaussian filter (weighted average with proximity)
        # This smoothes the image with a Gaussian kernel
        from scipy import ndimage
        
        # Try different sigma values for different smoothing levels
        filtered_gaussian_small = ndimage.gaussian_filter(left_half, sigma=1.0)  # Less smoothing
        gaussian_max_small = np.max(filtered_gaussian_small)
        
        filtered_gaussian_medium = ndimage.gaussian_filter(left_half, sigma=2.0)  # Medium smoothing
        gaussian_max_medium = np.max(filtered_gaussian_medium)
        
        filtered_gaussian_large = ndimage.gaussian_filter(left_half, sigma=3.0)  # More smoothing
        gaussian_max_large = np.max(filtered_gaussian_large)
        
        # Create a simple name for the plot (remove .png and _scale part)
        simple_name = re.match(r"^(\d+v\d+)_scale[\d\.]+\.png$", file_name).group(1)
        
        return simple_name, max_value, gaussian_max_small, gaussian_max_medium, gaussian_max_large
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return file_name, None

def main():
    # Current directory
    current_dir = os.getcwd()
    
    # Find all files matching the pattern
    pattern = r"^\d+v\d+_scale[\d\.]+\.png$" # More reliable pattern
    matching_files = []
    
    for file in os.listdir(current_dir):
        if re.match(pattern, file):
            matching_files.append(os.path.join(current_dir, file))
    
    if not matching_files:
        print(f"No files matching pattern '{pattern}' found in {current_dir}")
        return
    
    print(f"Found {len(matching_files)} matching files")
    
    # Sort files with intercalated ordering: each digit's self-comparison followed by its cross-comparisons
    def intercalated_sort_key(filename):
        match = re.match(r"^(\d+)v(\d+)_scale[\d\.]+\.png$", os.path.basename(filename))
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            # Group by first digit, then sort by whether it's a self-comparison
            # This keeps 5v5 and 5v2 together, then 1v1 and 1v2 together, etc.
            return (x, 0 if x == y else 1, y)
        return (999, 0, 0)  # Default case, should come last
    
    matching_files.sort(key=intercalated_sort_key)
    
    # Process each file
    results = []
    for file_path in matching_files:
        simple_name, max_value, gauss_small, gauss_medium, gauss_large = process_image(file_path)
        results.append((simple_name, max_value, gauss_small, gauss_medium, gauss_large))
    
    # Print results in a table format
    print("\nResults:")
    print("-" * 120)
    print(f"{'Image':<10} | {'Raw Max':>12} | {'Gauss σ=1.0':>15} | {'Gauss σ=2.0':>15} | {'Gauss σ=3.0':>15}")
    print("-" * 120)
    
    for simple_name, max_value, gauss_small, gauss_medium, gauss_large in results:
        if max_value is not None:
            print(f"{simple_name:<10} | {max_value:>12} | {gauss_small:>15.2f} | {gauss_medium:>15.2f} | {gauss_large:>15.2f}")
        else:
            print(f"{simple_name:<10} | {'ERROR':>12} | {'N/A':>15} | {'N/A':>15} | {'N/A':>15}")
    
    print("-" * 120)
    
    # Create a new plot with image names on x-axis and only Gaussian filter with sigma=3.0
    if results:
        # Filter out any results with None values
        valid_results = [(name, max_val, g_small, g_medium, g_large) 
                         for name, max_val, g_small, g_medium, g_large in results if max_val is not None]
        
        if valid_results:
            # Extract data for plotting
            names = [result[0] for result in valid_results]  # Image names (e.g., "1v1", "1v2")
            gauss_large = [result[4] for result in valid_results]  # Gaussian filter (sigma=3.0) only
            
            # Create simple bar chart for sigma=3.0 results
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Use blue color gradient based on value
            colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(gauss_large)))
            bars = ax.bar(names, gauss_large, color=colors, width=0.6)
            
            # Add data labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add labels and title
            ax.set_title('Gaussian Filter (σ=3.0) Maximum Values', fontsize=16)
            ax.set_xlabel('Image Pattern', fontsize=14)
            ax.set_ylabel('Pixel Value', fontsize=14)
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Add grid lines for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            fig.savefig("gaussian_sigma3_values.png", dpi=150)
            print("\nPlot saved as 'gaussian_sigma3_values.png'")
            plt.show()
        else:
            print("\nNo valid data points for plotting.")
    else:
        print("\nNo data available for plotting")

if __name__ == "__main__":
    main()