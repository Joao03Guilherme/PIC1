from __future__ import annotations

"""uc480_controller.py
Controller for the DCC1645C-HQ camera (UC480 driver).
THIS DRIVER ONLY WORKS WITH THIS CAMERA (or similar models).
Besides this, you must install the thorlabs dlls
"""

from typing import List, Optional, Tuple
import contextlib

import numpy as np
from pylablib.devices import uc480

__all__ = [
    "UC480Controller",
    "list_cameras",
]


def list_cameras() -> List[str]:
    """Return serial numbers of all UC480/uEye cameras detected by the driver."""
    return uc480.list_cameras(backend="uc480")


class UC480Controller:
    """High‑level wrapper around :class:`pylablib.devices.uc480.UC480Camera`.

    Parameters
    ----------
    serial
        Serial number of the camera to open. If *None*, the first camera found is used.
    init
        If *True* (default) the camera is opened immediately. Set to *False* if you
        want to modify instance attributes before opening the hardware.
    """

    # ------------------------------------------------------------------
    # Construction / teardown helpers
    # ------------------------------------------------------------------

    def __init__(self, serial: Optional[str] = None, *, init: bool = True):
        self.serial: str = self._pick_serial(serial)
        self._cam: Optional[uc480.UC480Camera] = None
        if init:
            self.open()

    # ----------------------- context manager -------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ----------------------- private helpers -------------------------

    @staticmethod
    def _pick_serial(user_serial: Optional[str]) -> str:
        cams = list_cameras()
        if not cams:
            raise RuntimeError("No UC480 cameras detected.")
        if user_serial is None:
            return cams[0]
        if user_serial not in cams:
            raise ValueError(
                f"Requested serial {user_serial} not among detected cameras: {cams}"
            )
        return user_serial

    def _require_open(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera is not open. Call `open()` first.")

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def open(self):
        """Open the connection to the hardware if not already open."""
        if self._cam is None:
            self._cam = uc480.UC480Camera(backend="uc480")
            self._cam.set_color_mode("mono8")

    def close(self):
        """Gracefully stop acquisition (if running) and close the connection."""
        if self._cam is not None:
            with contextlib.suppress(Exception):
                if self._cam.acquisition_in_progress():
                    self._cam.stop_acquisition()
            with contextlib.suppress(Exception):
                self._cam.close()
            self._cam = None

    # ------------------------------------------------------------------
    #  Camera‑wide information
    # ------------------------------------------------------------------

    @property
    def detector_size(self) -> Tuple[int, int]:
        """Return full sensor size as *(width, height)* in pixels."""
        self._require_open()
        return self._cam.get_detector_size()

    # ------------------------------------------------------------------
    #  Exposure & gain
    # ------------------------------------------------------------------

    def set_exposure(self, ms: float):
        """Set exposure time in **milliseconds**."""
        self._require_open()
        self._cam.set_exposure(ms)

    def get_exposure(self) -> float:
        """Current exposure time in milliseconds."""
        self._require_open()
        return self._cam.get_exposure()

    def set_gain(self, gain: float):
        """Set analogue gain (percentage, 0–100)."""
        self._require_open()
        self._cam.set_gain(gain)

    def get_gain(self) -> float:
        self._require_open()
        return self._cam.get_gain()

    # ------------------------------------------------------------------
    #  Region of Interest (ROI) and binning / subsampling
    # ------------------------------------------------------------------

    def reset_roi(self):
        """Set ROI to full sensor, binning 1×1."""
        w, h = self.detector_size
        self.set_roi(0, 0, w, h, hbin=1, vbin=1)

    def set_roi(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        hbin: int = 1,
        vbin: int = 1,
    ) -> None:
        """Define a rectangular *read‑out* window and optional binning factors.

        The hardware ROI is set such that the top‑left corner of the rectangle
        is at ``(x, y)`` and its size is ``width × height`` **before** binning.
        All values must be multiples of the camera’s increment (usually 1 or 2 px).
        """
        self._require_open()
        hstart = x
        hend = x + width
        vstart = y
        vend = y + height
        self._cam.set_roi(hstart, hend, vstart, vend, hbin, vbin)

    def get_roi(self) -> Tuple[int, int, int, int, int, int]:
        """Return *(hstart, hend, vstart, vend, hbin, vbin)* currently active."""
        self._require_open()
        return self._cam.get_roi()

    def set_subsampling(self, hsub: int = 1, vsub: int = 1):
        """Enable hardware subsampling (decimation) instead of binning."""
        self._require_open()
        self._cam.set_subsampling(hsub, vsub)

    # ------------------------------------------------------------------
    #  Acquisition helpers
    # ------------------------------------------------------------------

    def snap(self) -> np.ndarray:
        """Acquire one frame and return it as a NumPy array."""
        self._require_open()

        # stop any running continuous acquisition
        if self._cam.acquisition_in_progress():
            self._cam.stop_acquisition()

        # grab a single frame (pylablib built-in helper)
        frame = self._cam.snap()  # <-- use snap(), not single_frame()
        if frame is None:
            raise RuntimeError("Failed to retrieve frame – camera timeout?")
        return frame.copy()  # copy to detach from driver buffer


# ---------------------------------------------------------------------------
#  Demo / self‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with UC480Controller() as cam:
        print("Connected camera serial:", cam.serial)
        print("Sensor size:", cam.detector_size)

        cam.set_exposure(11)  # ms
        cam.reset_roi()  # full sensor, no binning

        frame = cam.snap()
        print("Captured frame shape:", frame.shape)

    plt.imshow(frame, cmap="gray")
    plt.title("UC480 single frame")
    plt.axis("off")
    plt.show()
