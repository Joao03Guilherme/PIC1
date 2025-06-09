#!/usr/bin/env python3
"""
optical_jtc_correlator_exulus.py
--------------------------------
Real-hardware binary / analog Joint-Transform Correlator driven by

  • **ExulusSLM**  - Python wrapper around Thorlabs EXULUS SDK  
  • **UC480Controller**  - IDS µEye / Thorlabs DCC1545M camera

It follows the same algorithmic flow as *binary_jtc_lightpipes.py*
but performs the two optical passes on real hardware.

Highlights
----------
* **checkerboard** keyword toggles a π-phase chequerboard on the input plane,
  mirroring the LightPipes simulation option.
* No plotting or GUI code – this class does *only* the correlation and returns
  metrics (optionally the raw camera planes) to the caller.
* Half-wave stroke is selected by default; calibration coefficients *m*, *b*
  (phase = m·grey + b) are passed straight to the Exulus driver.
"""
from __future__ import annotations

import time
import contextlib
from pathlib import Path
from typing import Tuple, Any

import numpy as np

# ------------- local hardware abstraction layers ----------------
from ..hardware.devices.Camera import UC480Controller          # camera
from ..hardware.devices.SLM import ExulusSLM         # SLM

# ----------------------------------------------------------------
EPS = 1e-6            # avoid divide-by-zero in normalisation
ANGLE_RANGE = np.pi      # use 0 to pi range for phase encoding

# ----------------------------------------------------------------
# Helper: ±1 checkerboard (5x5 pixel blocks)
# ----------------------------------------------------------------
def _checkerboard(shape: Tuple[int, int], block_size: int = 3) -> np.ndarray:
    height, width = shape
    pattern = np.ones(shape, dtype=float)
    
    # Calculate blocks in each dimension
    h_blocks = height // block_size
    w_blocks = width // block_size
    
    # Generate a block-level checkerboard pattern first
    for i in range(h_blocks + 1):  # +1 to handle edge case
        for j in range(w_blocks + 1):  # +1 to handle edge case
            # Determine if this block should be black or white
            # (in the checkerboard pattern, blocks are inverted when i+j is odd)
            if (i + j) % 2 == 1:
                # Calculate block boundaries (handling edge cases)
                y_start = i * block_size
                y_end = min((i + 1) * block_size, height)
                x_start = j * block_size
                x_end = min((j + 1) * block_size, width)
                
                # Set this block to -1
                pattern[y_start:y_end, x_start:x_end] = -1.0
    
    return pattern    # Values are +1.0 or -1.0

# ----------------------------------------------------------------
# Helper: side-by-side reference/object canvas (identical logic to sim)
# ----------------------------------------------------------------
def create_joint_input_plane(
    ref: np.ndarray,
    obj: np.ndarray,
    slm_shape: Tuple[int, int],
    thresh,
    scale: float = 0.05,
    binarize: bool = True,
) -> np.ndarray:
    rows, cols = slm_shape

    to255 = lambda a: np.zeros_like(a, np.uint8) if a.max() == 0 else (
        a / a.max() * 255).astype(np.uint8)
    ref255, obj255 = to255(ref), to255(obj)

    combo = np.hstack((ref255, obj255))          # e.g. 28×56 for MNIST
    h0, w0 = combo.shape
    fac = min(rows / h0, cols / w0) * scale
    w_new, h_new = max(1, int(w0 * fac)), max(1, int(h0 * fac))

    from PIL import Image
    combo_rs = Image.fromarray(combo).resize((w_new, h_new), Image.BICUBIC)

    canvas = np.zeros(slm_shape, float)
    y0, x0 = (rows - h_new) // 2, (cols - w_new) // 2
    canvas[y0:y0 + h_new, x0:x0 + w_new] = np.asarray(combo_rs, float)

    if binarize:
        thr = np.median(canvas) if thresh is None else thresh
        return np.where(canvas > thr, 1., -1.)
    return (canvas - canvas.min()) / (canvas.ptp() + EPS)


# ----------------------------------------------------------------
# Core class
# ----------------------------------------------------------------
class OpticalJTCorrelator:
    """
    Two-pass hardware Joint-Transform Correlator.

    Parameters
    ----------
    slm                 : existing ExulusSLM handle (or None → open new)
    cam                 : existing UC480Controller handle (or None → new)
    binary_input        : if True, ±1 encoding for the joint input
    binary_jps          : if True, binarise the JPS before second pass
    checkerboard        : if True, multiply input by ±1 chequerboard
    display_scale       : down-scale for digits on SLM (same as sim)
    sleep_time          : seconds to wait after each upload (exposure)
    blocking_fraction   : half-width of DC-blocking square relative to
                          shorter image dimension
    """

    def __init__(
        self,
        *,
        slm: ExulusSLM | None = None,
        cam: UC480Controller | None = None,
        binary_input: bool = True,
        binary_jps: bool = True,
        checkerboard: bool = False,
        display_scale: float = 0.05,
        sleep_time: float = 0.1,
        blocking_fraction: float = 0.005,
        exulus_kwargs: dict[str, Any] | None = None,
    ) -> None:

        self.slm = slm or ExulusSLM(**(exulus_kwargs or {}))
        self.cam = cam or UC480Controller()
        cam.reset_roi()

        self.binary_input = binary_input
        self.binary_jps = binary_jps
        self.checkerboard = checkerboard
        self.display_scale = display_scale
        self.sleep_time = sleep_time
        self.blocking_fraction = blocking_fraction

        self.w, self.h = self.slm.width, self.slm.height
        print(f"[OpticalJTC] SLM {self.w}×{self.h}  "
              f"binary_input={binary_input}  binary_jps={binary_jps}  "
              f"checkerboard={checkerboard}")

    # ------------- private wrappers ---------------------------------
    def _upload_and_snap(self, phase_img: np.ndarray) -> np.ndarray:
        # Calibrate the grey image
        grey_img = self.slm.phase_to_grey(phase_img)
        self.slm.display_grey(grey_img)
        time.sleep(self.sleep_time)
        return self.cam.snap().astype(np.float32)

    # ------------- public API ---------------------------------------
    def correlate(
        self,
        ref_digit: np.ndarray,
        obj_digit: np.ndarray,
        *,
        input_thresh=None,
        jps_thresh=None,
        return_planes: bool = False,
    ):
        """Return (peak, dc, (dy,dx)) or the full set if return_planes."""
        # 1. build joint input
        a0 = create_joint_input_plane(
            ref_digit, obj_digit, (self.h, self.w),
            thresh=input_thresh,
            scale=self.display_scale,
            binarize=self.binary_input,
        )
        if self.checkerboard:
            a0 *= _checkerboard((self.h, self.w))

        # map to 8-bit (0..127 (0...pi))
        if self.binary_input:
            a0_phase_map = (a0 + 1) / 2 * ANGLE_RANGE
        else:
            a0_phase_map = a0 * ANGLE_RANGE

        jps_raw = self._upload_and_snap(a0_phase_map)

        # 2. process JPS, second pass
        if self.binary_jps:
            thr = np.median(jps_raw) if jps_thresh is None else jps_thresh
            jps_phase = np.where(jps_raw > thr, 1.0, -1.0)
            jps_phase_map = (jps_phase + 1) / 2 * ANGLE_RANGE
        else:
            jps_norm = jps_raw / (jps_raw.max() + EPS)
            jps_phase_map = jps_norm * ANGLE_RANGE

        corr_plane = self._upload_and_snap(jps_phase_map)

        # 3. extract metrics
        cy, cx = np.array(corr_plane.shape) // 2
        dc_val = corr_plane.max()

        masked = corr_plane.copy()
        r = int(min(masked.shape) * self.blocking_fraction)
        masked[cy - r:cy + r + 1, cx - r:cx + r + 1] = 0.0

        peak_val = masked.max()
        py, px = np.unravel_index(masked.argmax(), masked.shape)
        dy, dx = py - cy, px - cx

        if return_planes:
            return peak_val, dc_val, (dy, dx), a0_phase_map, jps_phase_map, corr_plane, masked
        return peak_val, dc_val, (dy, dx)

    # ------------- context-manager sugar ----------------------------
    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.slm.close()
        with contextlib.suppress(Exception):
            self.cam.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()