"""
Python implementation of the Optical Joint Transform Correlator (JTC) algorithm.
This script displays two MNIST digits (a '1' and a '2') side-by-side on the SLM.
"""

from ..hardware.devices.SLM import SLMdisplay
from ..data.data import get_test_data # Added for MNIST
import time
import numpy as np
from scipy.ndimage import zoom # Added for resizing

slm = SLMdisplay(isImageLock=True)
resX, resY = slm.getSize()
print(f"SLM resolution: {resX}x{resY}")

# Target dimensions for the overall smaller display area on SLM (e.g., half SLM size)
overall_display_height = resY // 2
overall_display_width = resX // 2
print(f"Target overall display area for combined digits: {overall_display_width}x{overall_display_height}")

# Target dimensions for each individual MNIST image to fit side-by-side in this new smaller overall display area
mnist_target_height = overall_display_height # Each digit takes full height of the smaller display area
mnist_target_width = overall_display_width // 2 # Each digit takes half width of the smaller display area

print(f"Target dimensions for each MNIST image: {mnist_target_width}x{mnist_target_height}")

# 1. Load MNIST data
try:
    X_mnist, y_mnist = get_test_data(dataset='mnist')
    print("MNIST training data loaded.")
except Exception as e:
    print(f"Error loading MNIST data: {e}")
    slm.close()
    exit()

# 2. Get an image of a '1' and a '2'
img1_vec = None
img2_vec = None

idx_1 = np.where(y_mnist == 1)[0]
if len(idx_1) > 0:
    img1_vec = X_mnist[idx_1[0]]
else:
    print("Could not find an image of digit '1' in the dataset.")

idx_2 = np.where(y_mnist == 2)[0]
if len(idx_2) > 0:
    img2_vec = X_mnist[idx_2[0]]
else:
    print("Could not find an image of digit '2' in the dataset.")

if img1_vec is None or img2_vec is None:
    slm.close()
    exit()

# 3. Reshape original MNIST images (28x28)
img1_28x28 = img1_vec.reshape(28, 28)
img2_28x28 = img2_vec.reshape(28, 28)

# 4. Resize images to the new, smaller individual MNIST target dimensions
# Calculate zoom factors for individual MNIST images
zoom_y = mnist_target_height / 28.0
zoom_x = mnist_target_width / 28.0

print(f"Zoom factors for each MNIST image: zoom_y={zoom_y}, zoom_x={zoom_x}")

# Perform zoom, clip values to 0-255, round, and convert to uint8
resized_img1 = zoom(img1_28x28, (zoom_y, zoom_x), order=1)
resized_img1 = np.round(np.clip(resized_img1, 0, 255)).astype(np.uint8)

resized_img2 = zoom(img2_28x28, (zoom_y, zoom_x), order=1)
resized_img2 = np.round(np.clip(resized_img2, 0, 255)).astype(np.uint8)

print(f"Resized image 1 shape: {resized_img1.shape}") # Should be (mnist_target_height, mnist_target_width)
print(f"Resized image 2 shape: {resized_img2.shape}") # Should be (mnist_target_height, mnist_target_width)


# 5. Combine images side-by-side to form the smaller composite image
# Ensure actual resized dimensions are used, in case zoom results in slight variations
actual_mnist_h1, actual_mnist_w1 = resized_img1.shape
actual_mnist_h2, actual_mnist_w2 = resized_img2.shape

if actual_mnist_h1 != mnist_target_height or actual_mnist_w1 != mnist_target_width or \
   actual_mnist_h2 != mnist_target_height or actual_mnist_w2 != mnist_target_width:
    print("Warning: Resized MNIST image dimensions slightly differ from target after zoom.")
    print(f"Img1: {actual_mnist_w1}x{actual_mnist_h1}, Img2: {actual_mnist_w2}x{actual_mnist_h2}")
    # Update target dimensions to actual for combining, assuming heights are close enough
    # This uses the actual height of the first image and sum of actual widths
    mnist_target_height = actual_mnist_h1 # Or average, or min, depending on desired alignment
    # overall_display_width will be actual_mnist_w1 + actual_mnist_w2

small_combined_image = np.hstack((resized_img1, resized_img2))
combined_h, combined_w = small_combined_image.shape
print(f"Small combined image shape: {combined_w}x{combined_h}")

# 6. Create a full SLM resolution canvas (black background) and center the small_combined_image
final_slm_image = np.zeros((resY, resX), dtype=np.uint8)

# Calculate top-left coordinates for centering
start_row = (resY - combined_h) // 2
start_col = (resX - combined_w) // 2

# Calculate end coordinates
end_row = start_row + combined_h
end_col = start_col + combined_w

# Place the small combined image onto the center of the canvas
# Ensure the slice indices are within the bounds of final_slm_image and small_combined_image
if start_row >= 0 and start_col >= 0 and end_row <= resY and end_col <= resX:
    final_slm_image[start_row:end_row, start_col:end_col] = small_combined_image
else:
    # This case should ideally not be hit if overall_display_width/height are <= SLM res
    print("Error: Centered image placement is out of bounds. Placing at top-left instead.")
    # Fallback: place at top-left, cropping if necessary
    h_to_copy = min(combined_h, resY)
    w_to_copy = min(combined_w, resX)
    final_slm_image[0:h_to_copy, 0:w_to_copy] = small_combined_image[0:h_to_copy, 0:w_to_copy]

# 7. Display on SLM (this was step 6 in previous context, renumbered to 7)
print("Updating SLM with the centered, smaller combined MNIST digits...")
slm.updateArray(final_slm_image)
time.sleep(10) # Display for 5 seconds

print("Closing SLM.")
slm.close()
