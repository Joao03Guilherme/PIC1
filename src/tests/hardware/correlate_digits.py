"""
Simple script to run binary optical correlation between two MNIST digits.
Uses the OpticalJTCorrelator with binary mode and specific scale factor.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.data import get_test_data
from src.hardware.devices.SLM import SLMdisplay
from src.hardware.devices.Camera import UC480Controller
from src.distance.OpticalJTCorrelator import OpticalJTCorrelator

def run_correlation(ref_digit=1, obj_digit=2, slm_monitor=1):
    """
    Run optical correlation between two digits.
    
    Parameters:
    -----------
    ref_digit : int
        Reference digit (0-9)
    obj_digit : int
        Object digit to compare (0-9)
    slm_monitor : int
        Monitor number for SLM display
    """
    print(f"Loading MNIST data...")
    X_test, y_test = get_test_data(dataset_name="mnist")
    
    # Get digit images
    ref_idx = np.where(y_test == ref_digit)[0][0]
    obj_idx = np.where(y_test == obj_digit)[0][0]
    
    ref_image = X_test[ref_idx].reshape(28, 28) / 255.0
    obj_image = X_test[obj_idx].reshape(28, 28) / 255.0
    
    print(f"Initializing hardware...")
    # Initialize hardware
    slm = SLMdisplay(monitor=slm_monitor, isImageLock=True, alwaysTop=True)
    camera = UC480Controller()
    
    # Create correlator with binary mode and specified scale factor
    correlator = OpticalJTCorrelator(
        slm=slm,
        cam=camera,
        binary_input=True,         # Use binary input
        binary_jps=True,           # Use binary JPS
        display_scale_factor=0.01, # Use scale factor of 0.05
        sleep_time=0.1             # Time between optical passes
    )
    
    # Set ultra-short exposure
    correlator.set_exposure(0.001)  # 0.001 ms exposure
    print(f"Camera exposure set to 0.001ms")
    
    try:
        print(f"Running correlation between digit {ref_digit} and {obj_digit}...")
        # Run correlation
        peak_val, central_dc, peak_coords, corr_plane = correlator.correlate(ref_image, obj_image)
        
        # Calculate normalized peak
        similarity = peak_val / (central_dc + 1e-6)
        
        # Display results
        print(f"Results:")
        print(f"  Peak value: {peak_val:.4f}")
        print(f"  Central DC: {central_dc:.4f}")
        print(f"  Normalized peak: {similarity:.4f}")
        print(f"  Peak coordinates: {peak_coords}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Input images
        plt.subplot(131)
        plt.imshow(ref_image, cmap='gray')
        plt.title(f"Reference Digit {ref_digit}")
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(obj_image, cmap='gray')
        plt.title(f"Object Digit {obj_digit}")
        plt.axis('off')
        
        # Correlation plane
        plt.subplot(133)
        plt.imshow(corr_plane, cmap='hot')
        plt.title(f"Correlation Plane (Peak: {similarity:.4f})")
        plt.colorbar()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"correlation_{ref_digit}vs{obj_digit}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Saved result to {save_path}")
        
    finally:
        # Clean up resources
        print("Closing hardware resources...")
        correlator.close()

if __name__ == "__main__":
    # Define which digits to correlate
    ref_digit = 1  # Change as needed
    obj_digit = 2  # Change as needed
    slm_monitor = 1  # Change to your SLM monitor number
    
    # Run correlation
    run_correlation(ref_digit, obj_digit, slm_monitor)