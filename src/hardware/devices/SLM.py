# exulus_adapter.py  ─────────────────────────────────────────────
# Drop-in replacement for Thorlabs’ ExulusSLM based on slmPy
#
# Requires:
#   • the original slmPy classes (SLMdisplay, etc.) to be importable
#   • numpy ≥ 1.20
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
import warnings
import numpy as np

try:
    from slmPy import SLMdisplay        # ← your first code base
except ImportError as exc:
    raise ImportError(
        "Cannot import SLMdisplay from slmPy.  "
        "Ensure the slmPy package / folder is on PYTHONPATH."
    ) from exc


class ExulusSLM:
    """
    Thin wrapper that mimics Thorlabs’ ExulusSLM API using *slmPy*.

    Parameters
    ----------
    monitor : int, default 1
        Physical monitor index for the SLM window (0 = primary).
    isImageLock : bool, default False
        Whether to wait for the previous frame to finish drawing.
    alwaysTop : bool, default False
        If True the SLM window is kept top-most.
    m, b : float
        Calibration coefficients so that
            phase [rad] = m * grey + b
        (defaults match the Thorlabs sample).
    stroke_mode : {"half", "full"}, default "half"
        π-stroke or 2π-stroke.  slmPy has no equivalent, so the value
        is remembered but not applied to hardware.
    save_dir : str or Path, optional
        If given, every buffer sent to the SLM is saved as a PNG here.
        (Handy for debugging and synchronisation.)
    """

    # ───── initialisation & tear-down ──────────────────────────
    def __init__(
        self,
        *,
        monitor: int = 1,
        isImageLock: bool = False,
        alwaysTop: bool = False,
        m: float = 0.0174,
        b: float = 0.3639,
        stroke_mode: str = "half",
        save_dir: str | os.PathLike | None = None,
    ) -> None:

        # 1. bring up the SLM window
        self._slm = SLMdisplay(
            monitor=monitor, isImageLock=isImageLock, alwaysTop=alwaysTop
        )
        self.width, self.height = self._slm.getSize()

        # 2. calibration & mode bookkeeping
        self.m = float(m)
        self.b = float(b)
        self._stroke_mode = None
        self.set_stroke_mode(stroke_mode)

        # 3. optional frame capture
        if save_dir is not None:
            self._save_dir = Path(save_dir).expanduser()
            self._save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = None

        print(
            f"[Exulus] virtual SLM opened  ({self.width}×{self.height})  "
            f"stroke={self._stroke_mode}"
        )

    def close(self) -> None:
        """Gracefully release resources."""
        self._slm.close()
        print("[Exulus] closed")

    # ───── public API (matches Thorlabs) ───────────────────────
    def phase_to_grey(self, phase_rad: np.ndarray) -> np.ndarray:
        """Phase (rad) → 8-bit greyscale image."""
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
        """
        “half” → π-stroke, “full” → 2π-stroke.

        slmPy cannot change liquid-crystal stroke; the value is stored
        for parity with Thorlabs’ API and printed as a warning once.
        """
        mode = mode.lower()
        if mode.startswith("half"):
            self._stroke_mode = "half"
        elif mode.startswith("full"):
            self._stroke_mode = "full"
        else:
            raise ValueError("stroke_mode must be 'half' or 'full'")

        warnings.filterwarnings("once", category=UserWarning)
        warnings.warn(
            "stroke_mode is recorded but has no effect in the slmPy "
            "backend (this is a hardware feature).",
            RuntimeWarning,
            stacklevel=2,
        )

    # ───── internal helpers ────────────────────────────────────
    def _shape_guard(self, arr: np.ndarray) -> np.ndarray:
        """Ensure the array matches the SLM shape; pad/centre if needed."""
        if arr.shape == (self.height, self.width):
            return arr

        padded = np.zeros((self.height, self.width), dtype=arr.dtype)
        # centre (crop if larger)
        y0 = max(0, (self.height - arr.shape[0]) // 2)
        x0 = max(0, (self.width  - arr.shape[1]) // 2)
        ys = slice(0, min(arr.shape[0], self.height))
        xs = slice(0, min(arr.shape[1], self.width))
        padded[y0 : y0 + ys.stop, x0 : x0 + xs.stop] = arr[ys, xs]
        print(f"[Exulus] resized input {arr.shape} → {(self.height, self.width)}")
        return padded

    def _send(self, img_u8: np.ndarray) -> None:
        """Push the buffer to slmPy’s display and optionally save a copy."""
        # 1. optional PNG dump
        if self._save_dir is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self._save_dir / f"slm_image_{ts}.png"
            try:
                import matplotlib.pyplot as plt

                plt.imsave(filename, img_u8, cmap="gray")
                print(f"[Exulus] image saved to {filename}")
            except Exception as exc:
                print(f"[Exulus] failed to save {filename}: {exc}")

        # 2. show the image
        self._slm.updateArray(img_u8, sleep=0.0)

    # ───── convenience aliases (optional) ──────────────────────
    # keep parity with original Exulus driver attributes
    @property
    def dev(self):
        """Dummy attribute to satisfy code expecting .dev."""
        return None

    @property
    def _win(self):
        """Dummy attribute to satisfy code expecting ._win."""
        return None
