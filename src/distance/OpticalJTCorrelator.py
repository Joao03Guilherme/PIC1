"""
Python implementation of the Optical Joint Transform Correlator (JTC) algorithm.
This script displays two MNIST digits (a '1' and a '2') side-by-side on the SLM.
"""

from ..hardware.devices.SLM import SLMdisplay
from ..data.data import get_train_data # Added for MNIST
import time
import numpy as np
from scipy.ndimage import zoom # Added for resizing

slm = SLMdisplay(isImageLock=True)
resX, resY = slm.getSize()
print(f"SLM resolution: {resX}x{resY}")

# Target dimensions for each MNIST image to be displayed on half the SLM
target_height = resY
target_width = resX // 2

print(f"Target dimensions for each MNIST image: {target_width}x{target_height}")

# 1. Load MNIST data
try:
    X_mnist, y_mnist = get_train_data(dataset='mnist')
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

# 4. Resize images
# Calculate zoom factors
zoom_y = target_height / 28.0
zoom_x = target_width / 28.0

print(f"Zoom factors: zoom_y={zoom_y}, zoom_x={zoom_x}")

# Perform zoom, clip values to 0-255, round, and convert to uint8
resized_img1 = zoom(img1_28x28, (zoom_y, zoom_x), order=1) # order=1 for bilinear interpolation
resized_img1 = np.round(np.clip(resized_img1, 0, 255)).astype(np.uint8)

resized_img2 = zoom(img2_28x28, (zoom_y, zoom_x), order=1)
resized_img2 = np.round(np.clip(resized_img2, 0, 255)).astype(np.uint8)

print(f"Resized image 1 shape: {resized_img1.shape}")
print(f"Resized image 2 shape: {resized_img2.shape}")


# 5. Combine images side-by-side
if resized_img1.shape[0] != target_height or resized_img1.shape[1] != target_width or \
   resized_img2.shape[0] != target_height or resized_img2.shape[1] != target_width:
    print("Error: Resized image dimensions are not as expected.")
    print(f"Expected: ({target_height}, {target_width})")
    slm.close()
    exit()
    
combined_image = np.hstack((resized_img1, resized_img2))
print(f"Combined image shape: {combined_image.shape}")

if combined_image.shape[0] != resY or combined_image.shape[1] != resX:
    print(f"Warning: Combined image shape {combined_image.shape} does not match SLM resolution {resX}x{resY}.")

    final_slm_image = np.zeros((resY, resX), dtype=np.uint8)
    place_h = min(resY, combined_image.shape[0])
    place_w = min(resX, combined_image.shape[1])
    final_slm_image[0:place_h, 0:place_w] = combined_image[0:place_h, 0:place_w]
    combined_image = final_slm_image


# 6. Display on SLM
print("Updating SLM with the combined MNIST digits...")
slm.updateArray(combined_image)
time.sleep(10) # Display for 5 seconds

print("Closing SLM.")
slm.close()
