"""
Thorlabs CMOS Camera - Minimal Controller
=========================================

This driver is based on **pylablib**'s wrapper around the Thorlabs *TL Camera* SDK
(`pylablib.devices.Thorlabs.tlcam`). It provides a simplified, high-level API
for single-shot image acquisition.

Prerequisites
-------------
1. Install pylablib (≥ 1.4.2)::

       pip install pylablib

2. Install Thorlabs *TL Camera* SDK **and** make sure its DLL/SO can be found.
   The path can be supplied at runtime with::

       import pylablib as pll
       pll.par["devices/dlls/thorlabs_tlcam"] = r"C:\\Program Files\\Thorlabs\\Scientific Imaging\\DCx TL Cameras\\tlcam.dll"

   or (Linux)::

       pll.par["devices/dlls/thorlabs_tlcam"] = "/usr/local/lib/libtlcam.so"

3. Connect the camera and **disable** any other application that might already
   be talking to the device (ThorCam, μManager, LabVIEW, etc.).

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import numpy as np
from pylablib.devices.Thorlabs import tlcam as _tl


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def list_cameras() -> List[str]:
    """Return a list of serial numbers of all TL‑compatible cameras present."""
    return _tl.list_cameras()


def _auto_pick_serial(serial: Optional[str]) -> str:
    if serial is None:
        cams = list_cameras()
        if not cams:
            raise RuntimeError("No Thorlabs TL cameras were detected on this system.")
        serial = cams[0]
    return serial


# -----------------------------------------------------------------------------
# Main camera class
# -----------------------------------------------------------------------------


class ThorlabsCamera:
    """Simplified driver for Thorlabs CMOS cameras, focused on snapshots.

    Examples
    --------
    >>> with ThorlabsCamera() as cam:
    ...     cam.set_exposure(0.01)      # 10 ms
    ...     frame = cam.snap()           # numpy.ndarray, shape (H, W)
    ...     print(f"Captured frame of shape: {frame.shape}")
    """

    # ---------------------------------------------------------------------
    # Construction & context‑management
    # ---------------------------------------------------------------------

    def __init__(self, serial: Optional[str] = None, *, init: bool = True):
        self.serial: str = _auto_pick_serial(serial)
        self._cam: Optional[_tl.TLCamera] = None
        if init:
            self.open()

    # Context‑manager helpers ------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # Connection handling ----------------------------------------------------

    def open(self) -> None:
        """Open the underlying SDK connection (if not open already)."""
        if self._cam is None:
            self._cam = _tl.TLCamera(self.serial)
            # Optional: self._cam.set_timeout(2000)  # ms, for operations

    def close(self) -> None:
        """Close the connection and free resources."""
        if self._cam is not None:
            with contextlib.suppress(Exception): # Try to close gracefully
                if self._cam.is_acquiring():
                    self._cam.stop_acquisition()
            with contextlib.suppress(Exception):
                self._cam.close()
            self._cam = None

    # ---------------------------------------------------------------------
    # Property helpers
    # ---------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera is not opened. Call `open()` first.")

    # --- Exposure ----------------------------------------------------------
    def set_exposure(self, seconds: float) -> None:
        """Set camera exposure time in seconds."""
        self._require_open()
        self._cam.set_exposure(seconds)

    def get_exposure(self) -> float:
        """Get current camera exposure time in seconds."""
        self._require_open()
        return self._cam.get_exposure()

    # --- Sensor Size -------------------------------------------------------
    @property
    def sensor_size(self) -> Tuple[int, int]:
        """Get the full sensor size (width, height) in pixels."""
        self._require_open()
        return self._cam.get_sensor_size()

    # ---------------------------------------------------------------------
    # Acquisition helpers
    # ---------------------------------------------------------------------

    def snap(self) -> np.ndarray:
        """Acquire a *single* frame and return it as a NumPy array.
        The image data type is typically uint8 or uint16 depending on the camera model and settings.
        """
        self._require_open()
        # Ensure not in continuous acquisition mode from a previous state if an error occurred
        if self._cam.is_acquiring():
            self._cam.stop_acquisition()
        
        with self._cam.single_frame(): # Sets up for single frame, acquires, and stops
            frame = self._cam.get_last_frame()
        if frame is None:
            # This case should ideally be handled by pylablib raising an error,
            # but as a fallback:
            raise RuntimeError("Failed to acquire frame from camera.")
        return frame.copy() # Return a copy to avoid issues with buffer reuse


# -----------------------------------------------------------------------------
# Self‑test / demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with ThorlabsCamera() as cam:
        cam.set_exposure(0.02)  # 20 ms
        print("Sensor size:", cam.sensor_size)

        # Single‑shot --------------------------------------------------------
        img = cam.snap()
        plt.title("Single frame")
        plt.imshow(img, cmap="gray")
        plt.show()
