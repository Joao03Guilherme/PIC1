#!/usr/bin/env python3
"""
Find maximum value in the left half of various PNG images.

This script:
1. Searches for all files matching the pattern "[number]v[number]_scale0.8.png"
2. Loads each image as an 8-bit integer array
3. Divides the image in half (left and right)
4. Calculates and prints the maximum value in the left half
"""

import os
import glob
import numpy as np
from PIL import Image
import re

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
        (file_name, max_value_left_half)
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
        
        # Find maximum value in left half
        max_value = np.max(left_half)
        
        return file_name, max_value
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return file_name, None

def main():
    # Current directory
    current_dir = os.getcwd()
    
    # Find all files matching the pattern
    pattern = r"[0-9]+v[0-9]+_scale0\.8\.png"
    matching_files = []
    
    for file in os.listdir(current_dir):
        if re.match(pattern, file):
            matching_files.append(os.path.join(current_dir, file))
    
    if not matching_files:
        print(f"No files matching pattern '{pattern}' found in {current_dir}")
        return
    
    print(f"Found {len(matching_files)} matching files")
    
    # Sort files numerically by the x and y values in the filename
    def extract_xy(filename):
        match = re.match(r"([0-9]+)v([0-9]+)_scale0\.8\.png", os.path.basename(filename))
        if match:
            return int(match.group(1)), int(match.group(2))
        return 0, 0
    
    matching_files.sort(key=extract_xy)
    
    # Process each file
    results = []
    for file_path in matching_files:
        file_name, max_value = process_image(file_path)
        results.append((file_name, max_value))
    
    # Print results in a table format
    print("\nResults:")
    print("-" * 50)
    print(f"{'Filename':<25} | {'Max Value in Left Half':>20}")
    print("-" * 50)
    
    for file_name, max_value in results:
        if max_value is not None:
            print(f"{file_name:<25} | {max_value:>20}")
        else:
            print(f"{file_name:<25} | {'ERROR':>20}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()