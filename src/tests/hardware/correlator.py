"""
Test script for the OpticalJTCorrelator hardware implementation.

This script loads MNIST digit images from the test set, initializes the OpticalJTCorrelator,
and performs optical correlation between digit pairs. It demonstrates the use of optical
hardware for image correlation and compares different digits.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from datetime import datetime

# Add project root to sys.path to allow importing from project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data import get_test_data
from src.hardware.devices.SLM import SLMdisplay
from src.hardware.devices.Camera import UC480Controller
from src.distance.OpticalJTCorrelator import OpticalJTCorrelator


def get_sample_digits(digit1=1, digit2=2):
    """
    Get one sample of digit1 and one sample of digit2 from MNIST test set.
    
    Parameters:
    -----------
    digit1, digit2 : int
        The digit values to select (0-9)
        
    Returns:
    --------
    tuple
        (digit1_image, digit2_image, shape)
        where digit*_image are the flat vectors, and shape is the image shape (H, W)
    """
    X_test, y_test = get_test_data(dataset_name="mnist")
    print(f"Test data loaded, shape: {X_test.shape}")
    
    # Find first occurrence of digit1
    idx1 = np.where(y_test == digit1)[0]
    if len(idx1) == 0:
        raise RuntimeError(f"No instances of digit {digit1} found in test set")
    digit1_vec = X_test[idx1[0]].copy()
    
    # Find first occurrence of digit2
    idx2 = np.where(y_test == digit2)[0]
    if len(idx2) == 0:
        raise RuntimeError(f"No instances of digit {digit2} found in test set")
    digit2_vec = X_test[idx2[0]].copy()
    
    # MNIST is known to be 28x28, but we'll calculate it anyway
    # to be robust to other datasets
    n_features = X_test.shape[1]
    h_candidate = int(np.sqrt(n_features))
    while n_features % h_candidate != 0 and h_candidate > 0:
        h_candidate -= 1
    H = h_candidate
    W = n_features // H
    
    print(f"Selected digits: {digit1} at index {idx1[0]} and {digit2} at index {idx2[0]}")
    print(f"Image shape: {H}x{W}")
    
    return digit1_vec, digit2_vec, (H, W)


def plot_images_and_correlation(digit1_vec, digit2_vec, shape, similarity, correlation_image, distance, shift, digit1_label, digit2_label):
    """
    Plot the two input digit images and their correlation plane with detailed metrics.
    
    Parameters:
    -----------
    digit1_vec, digit2_vec : array_like
        Flattened image vectors
    shape : tuple(int, int)
        Image shape (H, W)
    similarity : float
        Correlation similarity metric
    correlation_image : array_like
        2D correlation plane image
    distance : float
        Distance metric (1/similarity)
    shift : tuple(int, int)
        (dy, dx) pixel shift of correlation peak
    digit1_label, digit2_label : int
        Labels of the digits being correlated
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout for plots
    gs = fig.add_gridspec(2, 6)
    
    # Plot the input images (larger on top row)
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.imshow(digit1_vec.reshape(shape), cmap='gray')
    ax1.set_title(f"Digit {digit1_label}", fontsize=14)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 3:])
    ax2.imshow(digit2_vec.reshape(shape), cmap='gray')
    ax2.set_title(f"Digit {digit2_label}", fontsize=14)
    ax2.axis('off')
    
    # Plot correlation results
    ax3 = fig.add_subplot(gs[1, 1:5])
    im = ax3.imshow(correlation_image, cmap='viridis')
    ax3.set_title(f"Optical Correlation Plane", fontsize=14)
    plt.colorbar(im, ax=ax3)
    
    # Add metrics as text
    textstr = '\n'.join((
        f"Similarity: {similarity:.4f}",
        f"Distance: {distance:.4f}",
        f"Peak Shift: ({shift[0]}, {shift[1]})"
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    # Set overall title
    fig.suptitle(f"Optical JTC Correlation: Digit {digit1_label} vs Digit {digit2_label}", 
                 fontsize=16, y=0.98)
    
    # Save figure with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(project_root, 
                            f'src/tests/hardware/optical_correlation_{digit1_label}vs{digit2_label}_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved correlation plot to: {save_path}")
    
    plt.show()


def run_correlation_test(digit1, digit2, correlator, verbose=True):
    """
    Run a single correlation test between two digits.
    
    Parameters:
    -----------
    digit1, digit2 : int
        The digits to correlate (0-9)
    correlator : OpticalJTCorrelator
        Initialized correlator object
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    tuple
        (digit1_vec, digit2_vec, shape, distance, shift, similarity, corr_plane)
    """
    # Get digit images
    digit1_vec, digit2_vec, shape = get_sample_digits(digit1=digit1, digit2=digit2)
    
    # Perform correlation
    if verbose:
        print(f"Running optical correlation between digit {digit1} and {digit2}...")
    
    t_start = datetime.now()
    distance, shift, similarity, corr_plane = correlator.correlate(
        digit1_vec, 
        digit2_vec, 
        shape=shape
    )
    elapsed = (datetime.now() - t_start).total_seconds()
    
    if verbose:
        print(f"Correlation completed in {elapsed:.2f} seconds")
        print(f"Results: Distance={distance:.4f}, Similarity={similarity:.4f}, Shift=({shift[0]}, {shift[1]})")
    
    return (digit1_vec, digit2_vec, shape, distance, shift, similarity, corr_plane)


def run_multi_correlation(correlator, digits_to_test=None, plot_all=False):
    """
    Run correlation tests between multiple digit pairs and display a comparison.
    
    Parameters:
    -----------
    correlator : OpticalJTCorrelator
        Initialized correlator object
    digits_to_test : list of tuples
        List of (digit1, digit2) pairs to test
    plot_all : bool
        Whether to plot individual correlation results
        
    Returns:
    --------
    dict
        Dictionary of correlation results
    """
    if digits_to_test is None:
        # Default: test a few interesting pairs
        digits_to_test = [
            (1, 1),  # Same digit (should have high similarity)
            (1, 7),  # Similar digits
            (1, 0),  # Different digits
            (3, 8),  # Visually similar digits
            (0, 8),  # Digits with holes
            (6, 9),  # Rotational relationship
        ]
    
    results = {}
    
    print(f"Running correlation tests for {len(digits_to_test)} digit pairs...")
    
    for i, (d1, d2) in enumerate(digits_to_test):
        print(f"Test {i+1}/{len(digits_to_test)}: Digit {d1} vs Digit {d2}")
        
        result = run_correlation_test(d1, d2, correlator, verbose=True)
        results[(d1, d2)] = result
        
        if plot_all:
            # Unpack the result
            digit1_vec, digit2_vec, shape, distance, shift, similarity, corr_plane = result
            
            # Plot this individual result
            plot_images_and_correlation(
                digit1_vec, digit2_vec, shape, similarity, corr_plane,
                distance, shift, d1, d2
            )
            
        # Small pause between tests
        sleep(1)
    
    # Create summary plot
    similarities = [results[pair][5] for pair in digits_to_test]
    distances = [results[pair][3] for pair in digits_to_test]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot similarities
    x_labels = [f"{d1} vs {d2}" for d1, d2 in digits_to_test]
    ax1.bar(x_labels, similarities, color='steelblue')
    ax1.set_title('Correlation Similarity by Digit Pair')
    ax1.set_ylabel('Similarity')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot distances
    ax2.bar(x_labels, distances, color='firebrick')
    ax2.set_title('Correlation Distance by Digit Pair')
    ax2.set_ylabel('Distance (1/Similarity)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the summary plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(project_root, 
                               f'src/tests/hardware/optical_correlation_summary_{timestamp}.png')
    plt.savefig(summary_path)
    print(f"Saved summary plot to: {summary_path}")
    plt.show()
    
    return results


def get_config():
    """Return default configuration settings"""
    # Default settings - modify these values directly in the code
    config = {
        "digit1": 1,              # First digit to correlate (0-9)
        "digit2": 2,              # Second digit to correlate (0-9)
        "monitor": 1,             # Monitor number for SLM display
        "exposure": 11,           # Camera exposure time in milliseconds
        "sleep_time": 0.05,        # Sleep time between optical passes (seconds) 
        "use_roi": False,          # Whether to set camera ROI
        "run_multi": False,       # Run multiple digit pairs
        "plot_all": False         # Plot all individual results in multi mode
    }
    
    return config


def main():
    """Main function to run optical correlation test"""
    # Get default configuration
    config = get_config()
    
    if config["run_multi"]:
        print("Starting OpticalJTCorrelator multi-digit test...")
    else:
        print(f"Starting OpticalJTCorrelator test with digits {config['digit1']} and {config['digit2']}...")
    
    correlator = None
    try:
        # Initialize hardware
        print(f"Initializing SLM (monitor {config['monitor']}) and camera...")
        slm = SLMdisplay(monitor=config['monitor'], isImageLock=True)
        camera = UC480Controller()
        
        # Set up the optical correlator
        print(f"Setting up OpticalJTCorrelator (sleep time: {config['sleep_time']}s)...")
        correlator = OpticalJTCorrelator(slm=slm, cam=camera, sleep_time=config['sleep_time'])
        
        # Set camera exposure
        correlator.set_exposure(config['exposure'])
        print(f"Camera exposure set to {config['exposure']}ms")
        
        # Optional: Set camera ROI
        if config['use_roi']:
            try:
                # Get sensor dimensions
                width, height = camera.detector_size
                print(f"Camera sensor size: {width}x{height}")
                
                # Center ROI at 50% of the full size
                correlator.set_roi(
                    x=width//4, 
                    y=height//4, 
                    width=width//2, 
                    height=height//2
                )
                print(f"Camera ROI set to center 50%")
            except Exception as e:
                print(f"Warning: Could not set ROI: {e}")
        else:
            print("Skipping ROI setup as requested")
        
        if config['run_multi']:
            # Run correlation tests between multiple digit pairs
            run_multi_correlation(correlator, plot_all=config['plot_all'])
        else:
            # Run a single correlation test
            digit1_vec, digit2_vec, shape, distance, shift, similarity, corr_plane = run_correlation_test(
                config['digit1'], config['digit2'], correlator
            )
            
            # Print detailed results for the single test
            print(f"Correlation details:")
            print(f"  Distance: {distance:.4f}")
            print(f"  Shift: ({shift[0]}, {shift[1]})")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Correlation plane shape: {corr_plane.shape}")
            
            # Plot and save results
            plot_images_and_correlation(
                digit1_vec, 
                digit2_vec, 
                shape, 
                similarity, 
                corr_plane,
                distance,
                shift,
                config['digit1'],
                config['digit2']
            )
        
    except Exception as e:
        print(f"Error during optical correlation test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up hardware resources
        if correlator:
            print("Closing hardware resources...")
            try:
                correlator.close()
                print("Hardware resources closed successfully")
            except Exception as e:
                print(f"Error closing hardware: {e}")


if __name__ == "__main__":
    main()