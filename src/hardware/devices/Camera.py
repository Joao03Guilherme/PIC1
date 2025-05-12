"""
Thorlabs DCC1645C CMOS Camera Driver
===================================

This driver is based on **pylablib**'s wrapper around the Thorlabs *TL Camera* SDK
(`pylablib.devices.Thorlabs.tlcam`). It provides a high-level, pythonic API
for single-shot or streaming acquisition, together with convenient helpers for
exposure, gain, ROI and external triggering.

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
import time
from typing import Iterable, List, Optional, Tuple

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

class DCC1645CCamera:
    """High-level driver for the **DCC1645C** CMOS camera.

    Examples
    --------
    >>> with DCC1645CCamera() as cam:
    ...     cam.set_exposure(0.01)      # 10 ms
    ...     frame = cam.snap()           # numpy.ndarray, shape (H, W)
    ...
    ...     cam.start_streaming()
    ...     for i, img in enumerate(cam.stream_frames(limit=100)):
    ...         process(img)
    ...     cam.stop_streaming()
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

    def __enter__(self):  # noqa: D401 – PEP‑343
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        self.close()

    # Connection handling ----------------------------------------------------

    def open(self) -> None:
        """Open the underlying SDK connection (if not open already)."""
        if self._cam is None:
            self._cam = _tl.TLCamera(self.serial)
            # optional: self._cam.set_timeout(2000)  # ms

    def close(self) -> None:
        """Close the connection and free resources."""
        if self._cam is not None:
            with contextlib.suppress(Exception):
                self._cam.close()
            self._cam = None

    # ---------------------------------------------------------------------
    # Property helpers
    # ---------------------------------------------------------------------

    # --- Exposure ----------------------------------------------------------
    def set_exposure(self, seconds: float) -> None:
        self._require_open()
        self._cam.set_exposure(seconds)

    def get_exposure(self) -> float:
        self._require_open()
        return self._cam.get_exposure()

    # --- Gain --------------------------------------------------------------
    def set_gain(self, gain: float) -> None:
        self._require_open()
        self._cam.set_gain(gain)

    def get_gain(self) -> float:
        self._require_open()
        return self._cam.get_gain()

    # --- Region of Interest ------------------------------------------------
    def set_roi(self, origin_x: int = 0, origin_y: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        """Set region of interest (values in **sensor** pixels).

        If *width* or *height* are *None*, the maximum size at the requested
        origin is used (i.e. full width/height from that point).
        """
        self._require_open()
        max_w, max_h = self.sensor_size
        if width is None:
            width = max_w - origin_x
        if height is None:
            height = max_h - origin_y
        self._cam.set_roi((origin_x, origin_y, width, height))

    @property
    def sensor_size(self) -> Tuple[int, int]:
        self._require_open()
        return self._cam.get_sensor_size()

    # --- Binning -----------------------------------------------------------
    def set_binning(self, hor: int = 1, ver: int = 1) -> None:
        """Set hardware binning factors (1 = no binning)."""
        self._require_open()
        self._cam.set_binning(hor, ver)

    def get_binning(self) -> Tuple[int, int]:
        self._require_open()
        return self._cam.get_binning()

    # --- Triggering --------------------------------------------------------
    def set_external_trigger(self, enable: bool = True, *, edge: str = "rising") -> None:
        """Enable/disable external trigger on the given edge ('rising'/'falling')."""
        self._require_open()
        mode = "external" if enable else "internal"
        self._cam.set_trigger(mode, edge=edge)

    # ---------------------------------------------------------------------
    # Acquisition helpers
    # ---------------------------------------------------------------------

    def snap(self) -> np.ndarray:
        """Acquire a *single* frame and return it as ``uint16`` Numpy array."""
        self._require_open()
        with self._cam.single_frame():
            frame = self._cam.get_last_frame()
        return frame.copy()

    # Continuous streaming --------------------------------------------------
    def start_streaming(self, *, buffer_size: int = 100, drop_frames: bool = True) -> None:
        """Start free‑running acquisition.

        Parameters
        ----------
        buffer_size
            Number of frames to keep in the driver ring‑buffer.
        drop_frames
            If *True*, the oldest frame is silently dropped when the buffer
            overflows. If *False*, acquisition stalls until frames are read.
        """
        self._require_open()
        self._cam.setup_continuous_acquisition(frames_per_buffer=buffer_size, drop_frames=drop_frames)
        self._cam.start_acquisition()

    def stop_streaming(self) -> None:
        self._require_open()
        self._cam.stop_acquisition()

    def stream_frames(self, *, limit: Optional[int] = None, timeout: float = 10.0) -> Iterable[np.ndarray]:
        """Yield frames in a *for* loop while streaming is running.

        Parameters
        ----------
        limit
            Stop after this many frames (*None* = unlimited).
        timeout
            Seconds to wait for a new frame before raising ``TimeoutError``.
        """
        self._require_open()
        if not self._cam.is_acquiring():
            raise RuntimeError("Stream not started. Call `start_streaming()` first.")

        count = 0
        start_time = time.monotonic()
        while limit is None or count < limit:
            if (time.monotonic() - start_time) > timeout:
                raise TimeoutError("Timeout while waiting for frames from camera.")
            if self._cam.get_frame_count() == 0:
                time.sleep(0.005)
                continue
            frame = self._cam.get_last_frame()
            count += 1
            yield frame.copy()

    # ---------------------------------------------------------------------
    # Utilities & sanity checks
    # ---------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._cam is None:
            raise RuntimeError("Camera is not opened. Call `open()` first.")

    # ---------------------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------------------

    def dump_info(self) -> None:
        """Print a short diagnostic block to stdout."""
        if self._cam is None:
            print("[DCC1645C] Not open")
            return
        print("[DCC1645C] Serial:", self.serial)
        print("  Sensor size: %dx%d px" % self.sensor_size)
        print("  ROI:        ", self._cam.get_roi())
        print("  Exposure:    %.6f s" % self.get_exposure())
        print("  Gain:        %.2f" % self.get_gain())
        print("  Binning:     %s×%s" % self.get_binning())
        print("  Trigger:     ", "external" if self._cam.get_trigger_mode() == "external" else "internal")


# -----------------------------------------------------------------------------
# Self‑test / demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with DCC1645CCamera() as cam:
        cam.set_exposure(0.02)  # 20 ms
        cam.set_gain(1.0)
        cam.dump_info()

        # Single‑shot --------------------------------------------------------
        img = cam.snap()
        plt.title("Single frame")
        plt.imshow(img, cmap="gray")
        plt.show()

        # Streaming ---------------------------------------------------------
        cam.start_streaming()
        frames = []
        for f in cam.stream_frames(limit=32):
            frames.append(f)
        cam.stop_streaming()
        print("Captured", len(frames), "frames in streaming mode.")