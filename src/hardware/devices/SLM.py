"""
Thorlabs **EXULUS** Phase‑Only SLM – Python Driver
=================================================

Thin, self‑contained wrapper for the Thorlabs *EXULUS* reflective LCOS spatial
light modulator.  It lets you push either a **phase map** (radians, 0→2π) or a
simple **8‑bit image** (uint8, 0→255) which is automatically scaled to a full
2π phase range.

Quick start
-----------
```python
import numpy as np
from exulus_driver import ExulusSLM

with ExulusSLM() as slm:
    # Send an 8‑bit grayscale image ---------------------
    img = np.random.randint(0, 256, slm.resolution, dtype=np.uint8)
    slm.show_image(img)

    slm.wait(2)

    # Send a true phase map (radians) -------------------
    H, W = slm.resolution
    y, x = np.indices((H, W))
    phase = np.mod(np.arctan2(y - H/2, x - W/2), 2*np.pi)
    slm.show_phase(phase)
    slm.wait(2)

    slm.blank()
```

Installation
------------
1. Run the *EXULUS* software installer (≥ v2.5.4).
2. In the SDK folder, execute **install_python_bindings.bat** (Windows) or
   install the wheel manually: `pip install thorlabs_exulus‑<ver>.whl`.
3. `pip install numpy`

No other packages are required.  If the official wheel is unavailable, this
file falls back to a minimal ctypes bridge to *ExulusSDK.dll* / *libExulusSDK.so*.

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import ctypes
import platform
import time
from typing import List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Try the official wheel first -------------------------------------------------
try:
    from thorlabs_exulus import Exulus as _exulus_api  # type: ignore
except ModuleNotFoundError:
    _exulus_api = None  # pyright: ignore[reportPrivateUsage]

# -----------------------------------------------------------------------------
# ctypes fallback --------------------------------------------------------------
# -----------------------------------------------------------------------------


def _build_ctypes_exulus() -> "_ExulusCAPI":
    """Create a *minimal* ctypes wrapper around the EXULUS SDK."""
    if platform.system() == "Windows":
        lib_names = ["ExulusSDK.dll"]
    elif platform.system() == "Darwin":
        lib_names = ["libExulusSDK.dylib"]
    else:
        lib_names = ["libExulusSDK.so"]

    for name in lib_names:
        try:
            _dll = ctypes.cdll.LoadLibrary(name)
            break
        except OSError:
            continue
    else:
        raise ImportError(
            "EXULUS SDK library not found. Install the official Python wheel or add the DLL/SO to PATH."
        )

    class _ExulusCAPI:  # minimal C‑API shim
        def __init__(self, dll):
            self._dll = dll
            # int exu_list_devices(char* serials, int max_len)
            self._dll.exu_list_devices.restype = ctypes.c_int
            self._dll.exu_list_devices.argtypes = [ctypes.c_char_p, ctypes.c_int]
            # void* exu_open(const char* serial)
            self._dll.exu_open.restype = ctypes.c_void_p
            self._dll.exu_open.argtypes = [ctypes.c_char_p]
            # void exu_close(void* handle)
            self._dll.exu_close.argtypes = [ctypes.c_void_p]
            # int exu_get_resolution(void* handle, int* w, int* h)
            self._dll.exu_get_resolution.restype = ctypes.c_int
            self._dll.exu_get_resolution.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            # int exu_write_image(void* handle, const uint16_t* data)
            self._dll.exu_write_image.restype = ctypes.c_int
            self._dll.exu_write_image.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint16),
            ]

        # ---------- helpers ----------
        def list_devices(self) -> List[str]:
            buf = ctypes.create_string_buffer(4096)
            n = self._dll.exu_list_devices(buf, len(buf))
            if n <= 0:
                return []
            return ctypes.string_at(buf).decode().split("\0")[:n]

        def open(self, serial: str):
            return self._dll.exu_open(serial.encode())

        def close(self, handle):
            self._dll.exu_close(handle)

        def get_resolution(self, handle) -> Tuple[int, int]:
            w = ctypes.c_int()
            h = ctypes.c_int()
            if (
                self._dll.exu_get_resolution(handle, ctypes.byref(w), ctypes.byref(h))
                != 0
            ):
                raise RuntimeError("exu_get_resolution failed")
            return h.value, w.value  # return (H, W)

        def write_image(self, handle, img16: np.ndarray):
            ptr = img16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            if self._dll.exu_write_image(handle, ptr) != 0:
                raise RuntimeError("exu_write_image failed")

    return _ExulusCAPI(_dll)


# Choose backend --------------------------------------------------------------
if _exulus_api is None:  # wheel missing -> use ctypes shim
    _exulus_api = _build_ctypes_exulus()  # type: ignore

# -----------------------------------------------------------------------------
# Main class ------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ExulusSLM:
    """High‑level wrapper for a single Thorlabs **EXULUS** SLM."""

    # ------------- construction/lifecycle -------------
    def __init__(self, serial: Optional[str] = None, *, auto_open: bool = True):
        self.serial = serial or self.list_devices()[0]
        self._handle = None  # type: ignore
        if auto_open:
            self.open()

    # ------ backend‑agnostic discovery ------
    @staticmethod
    def list_devices() -> List[str]:
        if hasattr(_exulus_api, "list_slms"):
            return _exulus_api.list_slms()  # official wheel
        return _exulus_api.list_devices()  # type: ignore

    # ------ open/close ------
    def open(self):
        if self._handle is not None:
            return
        if hasattr(_exulus_api, "Exulus"):
            self._handle = _exulus_api.Exulus(self.serial)
        else:
            self._handle = _exulus_api.open(self.serial)  # type: ignore

    def close(self):
        if self._handle is None:
            return
        if hasattr(_exulus_api, "Exulus"):
            with contextlib.suppress(Exception):
                self._handle.close()
        else:
            with contextlib.suppress(Exception):
                _exulus_api.close(self._handle)  # type: ignore
        self._handle = None

    # ------ context manager ------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------- properties -------------
    @property
    def resolution(self) -> Tuple[int, int]:  # (H, W)
        self._require_open()
        if hasattr(self._handle, "get_resolution"):
            w, h = self._handle.get_resolution()
            return h, w
        return _exulus_api.get_resolution(self._handle)  # type: ignore

    # ------------- main API -------------
    def show_phase(self, phase_map: np.ndarray):
        """Display a phase map (radians, range 0→2π)."""
        self._require_open()
        if phase_map.shape != self.resolution:
            raise ValueError("Phase map shape does not match the SLM resolution.")
        phase_norm = np.mod(phase_map, 2 * np.pi) / (2 * np.pi)
        img16 = np.round(phase_norm * 65535).astype(np.uint16)
        self._write(img16)

    def show_image(self, image: np.ndarray):
        """Display an 8‑bit image (uint8, 0–255 → 0–2π phase)."""
        if image.dtype != np.uint8:
            raise TypeError("`image` must be uint8 (0–255).")
        if image.shape != self.resolution:
            raise ValueError("Image shape does not match the SLM resolution.")
        img16 = image.astype(np.uint16) * 257  # 0…255 → 0…65535 (≈×257)
        self._write(img16)

    def blank(self):
        H, W = self.resolution
        self._write(np.zeros((H, W), dtype=np.uint16))

    def wait(self, seconds: float):
        time.sleep(seconds)

    # ------------- helpers -------------
    def _write(self, img16: np.ndarray):
        if hasattr(self._handle, "write_image"):
            self._handle.write_image(img16)
        else:
            _exulus_api.write_image(self._handle, img16)  # type: ignore

    def _require_open(self):
        if self._handle is None:
            raise RuntimeError("SLM not open. Call `open()` first.")

    # ------------- diagnostics -------------
    def dump_info(self):
        if self._handle is None:
            print("[ExulusSLM] not open")
            return
        H, W = self.resolution
        print(f"[ExulusSLM] Serial: {self.serial}")
        print(f"  Resolution:      {H}×{W}")


# -----------------------------------------------------------------------------
# Demo (run `python exulus_driver.py` to test) ---------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    with ExulusSLM() as slm:
        slm.dump_info()
        # Display checkerboard via show_image ---------------------------
        H, W = slm.resolution
        img = ((np.indices((H, W)).sum(axis=0) % 2) * 255).astype(np.uint8)
        slm.show_image(img)
        slm.wait(2)
        slm.blank()
