# exulus_driver.py
# ───────────────────────────────────────────────────────────────
# Lightweight Python interface for a Thorlabs EXULUS-HD SLM
# © 2025 – MIT licence
#
# Requirements
#   • Thorlabs EXULUS software 2.5 or newer
#       – EXULUS_COMMAND_LIB.py
#       – Thorlabs_EXULUS_CGHDisplay.py
#       – exulus_command_library.dll  (same folder / on PATH)
#   • numpy ≥ 1.20
# ───────────────────────────────────────────────────────────────
import ctypes
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------
# 1.  Load Thorlabs’ SDK wrappers
# ----------------------------------------------------------------
try:
    from . import EXULUS_COMMAND_LIB as ex               # device control
    from . import Thorlabs_EXULUS_CGHDisplay as disp      # display helper
except ImportError as exc:
    raise ImportError(
        "Thorlabs SDK modules not found.  Ensure the EXULUS SDK’s "
        "Python folder is on PYTHONPATH and the DLL is on PATH."
    ) from exc

# ----------------------------------------------------------------
# 1a.  Declare C-function prototypes (prevents access-violation)
# ----------------------------------------------------------------
disp.CghDisplayCreateWindow.argtypes = (
    ctypes.c_int,      # monitor index
    ctypes.c_int,      # width
    ctypes.c_int,      # height
    ctypes.c_char_p,   # ANSI title
)
disp.CghDisplayCreateWindow.restype  = ctypes.c_int

disp.CghDisplaySetWindowInfo.argtypes = (
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
disp.CghDisplaySetWindowInfo.restype  = ctypes.c_int

disp.CghDisplayShowWindow.argtypes = (
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8),
)
disp.CghDisplayShowWindow.restype  = ctypes.c_int

# ----------------------------------------------------------------
# 2.  Panel resolutions (pixels) – WUXGA for every HD model
# ----------------------------------------------------------------
_PANEL_SIZE = {
    "EXULUS-HD2":   (1920, 1200),
    "EXULUS-HD3":   (1920, 1200),
    "EXULUS-HD4":   (1920, 1200),
    "EXULUS-HDxHP": (3840, 2160),   # 4 K high-power head
    "EXULUS-4K1":   (3840, 2160),
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
        Select π- or 2π-stroke immediately after opening the device.
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
            serial, dev_type_raw = devs[device_index]
        except IndexError:
            raise ValueError(f"device_index {device_index} out of range.")

        # normalise dev_type to one of our table keys
        dev_type_raw = dev_type_raw.upper()
        if   "HD2" in dev_type_raw: dev_type = "EXULUS-HD2"
        elif "HD3" in dev_type_raw: dev_type = "EXULUS-HD3"
        elif "HD4" in dev_type_raw: dev_type = "EXULUS-HD4"
        elif "4K"  in dev_type_raw: dev_type = "EXULUS-4K1"
        else:                       dev_type = dev_type_raw  # hope for match

        # 2. open the chosen device
        hdl = ex.EXULUSOpen(serial, 115200, 5)
        if hdl < 0:
            raise RuntimeError(f"Failed to open EXULUS with serial {serial}")
        self.dev = hdl

        # 3. resolution lookup
        try:
            self.width, self.height = _PANEL_SIZE[dev_type]
        except KeyError:
            raise RuntimeError(f"Unknown EXULUS type '{dev_type}' – "
                               "add its resolution to _PANEL_SIZE.")

        # 4. set stroke mode
        self.set_stroke_mode(stroke_mode)

        # 5. create a full-screen, border-less window – try every monitor
        mon_cnt = disp.CghDisplayGetMonitorCount()
        if mon_cnt == 0:
            raise RuntimeError("CGH-Display DLL reports no monitors.")

        self._win = 0
        title = "EXULUS SLM"                              
        for mon_id in range(mon_cnt):
            try:
                win = disp.CghDisplayCreateWindow(
                    mon_id, self.width, self.height, title
                )
            except RuntimeError as exc:
                print(f"[Exulus] CghDisplayCreateWindow failed on monitor "
                      f"{mon_id}: {exc}")
                continue

            if win > 0:                                    # success
                self._win = win
                break

        if self._win <= 0:
            raise RuntimeError(
                f"Could not create {self.width}×{self.height} window on any "
                f"monitor 0–{mon_cnt-1}.  Match one monitor to that exact "
                "resolution (no scaling) and try again."
            )

        ret = disp.CghDisplaySetWindowInfo(
            self._win, self.width, self.height, 1  # 1 = 8-bit greyscale
        )
        if ret != 0:
            raise RuntimeError(
                f"CghDisplaySetWindowInfo failed (code {ret})."
            )

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
        """Phase (rad) → 8-bit greyscale."""
        grey = (phase_rad - self.b) / self.m
        return np.clip(np.rint(grey), 0, 255).astype(np.uint8)

    def grey_to_phase(self, img: np.ndarray) -> np.ndarray:
        """8-bit greyscale → phase (rad)."""
        return self.m * img.astype(float) + self.b

    def display_phase(self, phase_rad: np.ndarray) -> None:
        """Send a phase map to the SLM."""
        phase_rad = self._shape_guard(phase_rad)
        self._send(self.phase_to_grey(phase_rad))

    def display_grey(self, img: np.ndarray) -> None:
        """Send an 8-bit image (already γ-corrected) to the SLM."""
        img = self._shape_guard(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        self._send(img)

    def set_stroke_mode(self, mode: str) -> None:
        """“half” → π-stroke, “full” → 2π-stroke."""
        mode = mode.lower()
        if   mode.startswith("half"): ex.EXULUSSetPhaseStrokeMode(self.dev, 0x01)
        elif mode.startswith("full"): ex.EXULUSSetPhaseStrokeMode(self.dev, 0x00)
        else: raise ValueError("stroke_mode must be 'half' or 'full'")

    # ———————————————————————————————————————————————
    # internal helpers
    # ———————————————————————————————————————————————
    def _shape_guard(self, arr: np.ndarray) -> np.ndarray:
        """Ensure array matches the SLM shape; pad/centre if needed."""
        if arr.shape == (self.height, self.width):
            return arr
        padded = np.zeros((self.height, self.width), dtype=arr.dtype)
        # centre (crop if larger)
        y0 = max(0, (self.height - arr.shape[0]) // 2)
        x0 = max(0, (self.width  - arr.shape[1]) // 2)
        ys = slice(0, min(arr.shape[0], self.height))
        xs = slice(0, min(arr.shape[1], self.width))
        padded[y0:y0+ys.stop, x0:x0+xs.stop] = arr[ys, xs]
        print(f"Resized input {arr.shape} → {(self.height, self.width)}")
        return padded

    def _send(self, img_u8: np.ndarray) -> None:
        """Push the buffer to the CGH-Display window."""
        buf = np.ascontiguousarray(img_u8)
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        ret = disp.CghDisplayShowWindow(self._win, ptr)
        if ret != 0:
            raise RuntimeError(f"CghDisplayShowWindow failed (code {ret}).")
