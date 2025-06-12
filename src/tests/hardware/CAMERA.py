import sys
import os
import matplotlib.pyplot as plt

# Add project root to sys.path to allow importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.hardware.devices.Camera import UC480Controller, list_cameras

serials = list_cameras()  # e.g. ['4101859088']
if not serials:
    raise RuntimeError("No UC480 cameras found")

# Use UC480Controller with a context manager
try:
    with UC480Controller(serial=serials[0]) as cam:
        cam.set_exposure(11)  # 11 ms
        cam.reset_roi() # Good practice to ensure full ROI unless specified otherwise

        img = cam.snap()
        print(f"Captured frame with shape: {img.shape} using UC480Controller")

except Exception as e:
    print(f"An error occurred: {e}")
    img = None

if img is not None:
    plt.imshow(img, cmap="gray")
    plt.title("Single frame via UC480Controller")
    plt.show()
else:
    print("Failed to capture image.")
