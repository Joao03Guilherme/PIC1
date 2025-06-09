# exulus_driver.py
# ───────────────────────────────────────────────────────────────
# Lightweight Python interface for a Thorlabs Exulus-HD SLM
# © 2025 – MIT licence
#
# Requirements
# • Thorlabs EXULUS software 2.5 or newer
#   – EXULUS_COMMAND_LIB.py
#   – Thorlabs_EXULUS_CGHDisplay.py
#   – exulus_command_library.dll  (same folder / on PATH)
# • numpy ≥ 1.20
# ───────────────────────────────────────────────────────────────
from pathlib import Path
import ctypes
import numpy as np

# ----------------------------------------------------------------
# 1.  Load Thorlabs’ SDK wrappers
# ----------------------------------------------------------------
try:
    from . import EXULUS_COMMAND_LIB as ex      # device-control wrapper
    from . import Thorlabs_EXULUS_CGHDisplay as disp  # display helper
except ImportError as exc:
    raise ImportError(
        "Thorlabs SDK modules not found.  Ensure the EXULUS SDK’s "
        "Python folder is on PYTHONPATH and the DLL is on PATH."
    ) from exc


# ----------------------------------------------------------------
# 2.  Hard-coded panel resolutions (pixels)
#     SDK does not expose these programmatically.
# ----------------------------------------------------------------
_PANEL_SIZE = {
    "EXULUS-HD2":   (1920, 1080),
    "EXULUS-HD3":   (1920, 1200),
    "EXULUS-HD4":   (1920, 1080),
    "EXULUS-HDxHP": (3840, 2160),   # high-power version
}


# ----------------------------------------------------------------
# 3.  Helper class
# ----------------------------------------------------------------
class ExulusSLM:
    """
    Thin wrapper around the EXULUS Python SDK.

    Parameters
    ----------
    device_index : int
        0 for the first SLM returned by EXULUSListDevices().
    m, b : float
        Calibration coefficients so that
            phase [rad] = m * grey + b
        (half-wave calibration by default:  grey 127 → π).
    stroke_mode : {"half", "full"}
        Select π or 2π stroke immediately after opening the device.
    """

    # ———————————————————————————————————————————————
    # initialisation & tear-down
    # ———————————————————————————————————————————————
    def __init__(self, *, device_index=0, m=0.0174, b=0.3639,
                 stroke_mode="half") -> None:

        # 1. enumerate devices
        devs = ex.EXULUSListDevices()
        if not devs:
            raise RuntimeError("No EXULUS devices detected.")
        try:
            serial, dev_type = devs[device_index]
            if "HD3" in dev_type:
                dev_type = "EXULUS-HD3"
        except IndexError:
            raise ValueError(f"device_index {device_index} out of range.")

        # 2. open the chosen device
        hdl = ex.EXULUSOpen(serial, 115200, 5)
        if hdl < 0:
            raise RuntimeError(f"Failed to open EXULUS with serial {serial!s}")
        self.dev = hdl

        # 3. resolution lookup
        try:
            self.width, self.height = _PANEL_SIZE[dev_type]
        except KeyError:
            raise RuntimeError(f"Unknown EXULUS type '{dev_type}' – "
                               "add its resolution to _PANEL_SIZE.")

        # 4. set stroke mode
        self.set_stroke_mode(stroke_mode)

        # 5. create a full-screen border-less window on monitor #1
        mon_cnt = disp.CghDisplayGetMonitorCount()
        mon_id  = 1 if mon_cnt >= 2 else 0
        self._win = disp.CghDisplayCreateWindow(mon_id,
                                                self.width, self.height,
                                                "EXULUS SLM")
        
        if self._win <= 0:
            raise RuntimeError(f"Failed to create display window for {serial!s}")
        
        disp.CghDisplaySetWindowInfo(self._win,
                                     self.width, self.height,
                                     1)                 # 1 = greyscale

        # calibration parameters
        self.m = float(m)
        self.b = float(b)

        print(f"[Exulus] opened serial {serial}  "
              f"{self.width}×{self.height}  stroke={stroke_mode}")

    def close(self) -> None:
        """Gracefully release resources."""
        disp.CghDisplayCloseWindow(self._win)
        ex.EXULUSClose(self.dev)
        print("[Exulus] closed")

    # ———————————————————————————————————————————————
    # public API
    # ———————————————————————————————————————————————
    def phase_to_grey(self, phase_rad: np.ndarray) -> np.ndarray:
        """Map desired phase (rad) → 8-bit greyscale."""
        grey = (phase_rad - self.b) / self.m
        return np.clip(np.rint(grey), 0, 255).astype(np.uint8)

    def grey_to_phase(self, img: np.ndarray) -> np.ndarray:
        """8-bit greyscale → phase (rad)."""
        return self.m * img.astype(float) + self.b

    def display_phase(self, phase_rad: np.ndarray) -> None:
        """Send a *phase map* to the SLM."""
        phase_rad = self._shape_guard(phase_rad)
        img = self.phase_to_grey(phase_rad)
        self._send(img)

    def display_grey(self, img: np.ndarray) -> None:
        """Send an 8-bit image (already γ-corrected) to the SLM."""
        img = self._shape_guard(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        self._send(img)

    def set_stroke_mode(self, mode: str) -> None:
        """
        `"half"` → π-radian stroke  
        `"full"` → 2π-radian stroke
        """
        mode = mode.lower()
        if mode.startswith("half"):
            ex.EXULUSSetPhaseStrokeMode(self.dev, 0x01)
        elif mode.startswith("full"):
            ex.EXULUSSetPhaseStrokeMode(self.dev, 0x00)
        else:
            raise ValueError("stroke_mode must be 'half' or 'full'")

    # ———————————————————————————————————————————————
    # internal helpers
    # ———————————————————————————————————————————————
    def _shape_guard(self, arr: np.ndarray) -> np.ndarray:
        """Check array shape and pad/center if necessary."""
        if arr.shape != (self.height, self.width):
            # Create a centered padded array instead of raising an error
            padded_arr = np.zeros((self.height, self.width), dtype=arr.dtype)
            # Calculate the centering offsets
            y_offset = max(0, (self.height - arr.shape[0]) // 2)
            x_offset = max(0, (self.width - arr.shape[1]) // 2)
            # Calculate the region to copy (handle arrays larger than SLM too)
            y_slice = slice(0, min(arr.shape[0], self.height))
            x_slice = slice(0, min(arr.shape[1], self.width))
            y_target = slice(y_offset, y_offset + min(arr.shape[0], self.height))
            x_target = slice(x_offset, x_offset + min(arr.shape[1], self.width))
            # Copy the visible part of the input to the padded array
            padded_arr[y_target, x_target] = arr[y_slice, x_slice]
            print(f"Resized input array from {arr.shape} to SLM resolution {(self.height, self.width)}")
            return padded_arr
        return arr

    def _send(self, img_u8: np.ndarray) -> None:
        """Push the buffer to the already-created display window."""
        # ensure C-contiguous memory for ctypes
        buf = np.ascontiguousarray(img_u8)
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ret   = disp.CghDisplayShowWindow(self._win, buf_ptr)
        if ret != 0:
            raise RuntimeError(f"Failed to display image on SLM: {ret}")
