from pylablib.devices import uc480
import matplotlib.pyplot as plt

serials = uc480.list_cameras()  # e.g. ['4101859088']
if not serials:
    raise RuntimeError("No UC480 cameras found")

cam = uc480.UC480Camera(serials[0])  # open first camera
cam.set_exposure(11)  # 10 ms

img = cam.snap()
cam.close()

plt.imshow(img, cmap="gray")
plt.title("Single frame")
plt.show()
