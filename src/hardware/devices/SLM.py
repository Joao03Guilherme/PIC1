# exulus_driver.py
# ───────────────────────────────────────────────────────────────
# Lightweight Python interface for a Thorlabs Exulus-HD SLM
# © 2025 – released under the MIT licence
#
# Requirements
# • Thorlabs EXULUS software 2.5.x or newer
#   (installs  ExulusPython.dll  and  exulus.py)
# • numpy ≥ 1.20
# ───────────────────────────────────────────────────────────────
from pathlib import Path
import time
import numpy as np

# ----------------------------------------------------------------
# 1.  Load Thorlabs’ Python wrapper (exulus.py)
#    Adjust the path if you installed EXULUS to a custom folder.
# ----------------------------------------------------------------
try:
    import exulus                         # provided by Thorlabs
except ImportError as exc:
    raise ImportError(
        "exulus.py not found.  Install the Exulus SDK and make sure\n"
        "…\\Thorlabs\\EXULUS\\SDK\\Python is on PYTHONPATH."
    ) from exc


# ----------------------------------------------------------------
# 2. Helper class
# ----------------------------------------------------------------
class ExulusSLM:
    """
    Thin wrapper around the Exulus Python SDK.

    Parameters
    ----------
    device_index : int
        0 for the first (and usually only) SLM connected.
    m, b : float
        Calibration coefficients so that
            phase [rad] = m * grey + b
        (half-wave calibration by default:  grey 127 → π)
    stroke_mode : {"half", "full"}
        Select π or 2π stroke right after opening the device.
    """

    def __init__(self, *, device_index=0, m=0.0174, b=0.3639,
                 stroke_mode="half") -> None:

        self.dev = exulus.open_device(device_index)   # handle from SDK
        self.set_stroke_mode(stroke_mode)

        # SLM native width × height (HD3 = 1920×1152 user, 1920×1200 panel)
        self.width, self.height = exulus.get_size(self.dev)

        # linear calibration  phase = m·grey + b  (→ grey = (phase-b)/m)
        self.m = float(m)
        self.b = float(b)

        print(f"[Exulus] opened device {device_index}  "
              f"{self.width}×{self.height}  stroke = {stroke_mode}")

    # ———————————————————————————————————————————————
    # public API
    # ———————————————————————————————————————————————
    def phase_to_grey(self, phase_rad: np.ndarray) -> np.ndarray:
        """
        Map desired phase (radians) to an 8-bit image using the calibration.
        Values are clipped to [0,255].
        """
        grey = (phase_rad - self.b) / self.m
        return np.clip(np.rint(grey), 0, 255).astype(np.uint8)

    def grey_to_phase(self, img: np.ndarray) -> np.ndarray:
        """Inverse mapping: 8-bit image → phase map (radians)."""
        return self.m * img.astype(float) + self.b

    def display_phase(self, phase_rad: np.ndarray) -> None:
        """
        Send a *phase map* (float array, radians) to the SLM.
        Shape must match (height, width).
        """
        self._shape_guard(phase_rad)
        img = self.phase_to_grey(phase_rad)
        self._send(img)

    def display_grey(self, img: np.ndarray) -> None:
        """
        Send an 8-bit greyscale image straight to the SLM.
        Shape must match (height, width).
        """
        self._shape_guard(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        self._send(img)

    def set_stroke_mode(self, mode: str) -> None:
        """
        "half"  → π rad total stroke, LUT step ≈ π/255  
        "full"  → 2π rad stroke, LUT step ≈ 2π/255
        """
        mode = mode.lower()
        if mode.startswith("half"):
            exulus.set_phase_stroke_mode(self.dev, 0)
        elif mode.startswith("full"):
            exulus.set_phase_stroke_mode(self.dev, 1)
        else:
            raise ValueError("stroke_mode must be 'half' or 'full'")

    def close(self) -> None:
        exulus.close_device(self.dev)
        print("[Exulus] closed")

    # ———————————————————————————————————————————————
    # internal helpers
    # ———————————————————————————————————————————————
    def _shape_guard(self, arr: np.ndarray) -> None:
        if arr.shape != (self.height, self.width):
            raise ValueError(f"Array shape {arr.shape} "
                             f"does not match SLM resolution "
                             f"({self.height}, {self.width})")

    def _send(self, img_u8: np.ndarray) -> None:
        exulus.send_buffer(self.dev, img_u8)

