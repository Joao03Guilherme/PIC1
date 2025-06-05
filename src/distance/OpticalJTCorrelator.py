"""optical_jtc_correlator.py
--------------------------------------------------
Real‑hardware implementation of the *binary_jtc_lightpipes.py* demo.

► Uses the very same helper functions •create_joint_input_plane• and
  •perform_jtc_correlation• logic, but replaces LightPipes propagation
  with an SLM/Camera hardware loop.

► Requires two thin wrapper classes already provided elsewhere in the
  project:
    • UC480Controller  – IDS µEye / Thorlabs DCC1545M‑style camera
    • SLMdisplay       – Display‑port driven reflective LCoS SLM

The correlator works in two passes exactly like the simulated Joint
Transform Correlator (JTC):

 1.  Joint input plane (reference + object) is shown on the SLM, the
     camera records the raw Joint‑Power Spectrum (JPS).
 2.  The JPS is optionally binarised, then uploaded back to the SLM.
     A second camera exposure returns the correlation plane – from
     which the correlation peaks are extracted.

All amplitude values shown on the SLM are 8‑bit (0‑127) in order to fit
both positive and negative logic levels of the binary JTC on a
commercial 8‑bit display pipeline.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import contextlib

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------
# Local project imports (hardware abstraction layers)
# ------------------------------------------------------------------
from ..hardware.devices.Camera import UC480Controller  # noqa: E402, F401
from ..hardware.devices.SLM import SLMdisplay          # noqa: E402, F401

###############################################################################
# Constants & pure‑Python helpers (identical to binary_jtc_lightpipes.py)     #
###############################################################################

EPS = 1e-6  # avoid divide‑by‑zero

# ------------------------------------------------------------------
# create_joint_input_plane (VERBATIM from binary_jtc_lightpipes.py)
# ------------------------------------------------------------------

def create_joint_input_plane(
    digit_array_ref: np.ndarray,
    digit_array_obj: np.ndarray,
    slm_shape: Tuple[int, int],
    thresh,
    display_scale_factor: float = 0.2,
    binarize: bool = True,
) -> np.ndarray:
    """Return an SLM‑sized plane containing reference & object side‑by‑side.

    Parameters
    ----------
    digit_array_ref, digit_array_obj
        2‑D float arrays in range 0–1 (MNIST digits, for instance).
    slm_shape
        (rows, cols) of the physical SLM panel.
    thresh
        Manual threshold for binarisation; *None* → median of canvas.
    display_scale_factor
        Additional down‑scaling so the digits take only a fraction of the
        SLM area (allows wider reference/object spacing without
        clipping).
    binarize
        If *True* → output values in {‑1, +1}; else scaled 0–1.
    """
    slm_rows, slm_cols = slm_shape

    # --- helper: bring each digit to 0‑255 ---------------------------------
    def _scale_0_255(arr: np.ndarray) -> np.ndarray:
        m = arr.max()
        return (np.zeros_like(arr, np.uint8) if m == 0 else
                np.round(arr / m * 255).astype(np.uint8))

    ref_255 = _scale_0_255(digit_array_ref)
    obj_255 = _scale_0_255(digit_array_obj)

    # --- side‑by‑side canvas ----------------------------------------------
    combo = np.hstack((ref_255, obj_255))             # 28 × 56 for MNIST
    H0, W0 = combo.shape

    scale = min(slm_rows / H0, slm_cols / W0)
    Wf = int(max(1, W0 * scale * display_scale_factor))
    Hf = int(max(1, H0 * scale * display_scale_factor))

    combo_rs = Image.fromarray(combo).resize((Wf, Hf), Image.BICUBIC)

    canvas = np.zeros((slm_rows, slm_cols), float)
    y0, x0 = (slm_rows - Hf) // 2, (slm_cols - Wf) // 2
    canvas[y0 : y0 + Hf, x0 : x0 + Wf] = np.asarray(combo_rs, float)

    if binarize:
        t = np.median(canvas) if thresh is None else thresh
        return np.where(canvas > t, 1.0, -1.0)

    return (canvas - canvas.min()) / (canvas.ptp() + EPS)


# ------------------------------------------------------------------
# Helper: minimal padding so images always match SLM pixel matrix
# ------------------------------------------------------------------

def _pad_to_slm(img: np.ndarray, slm_shape: Tuple[int, int]) -> np.ndarray:
    """Zero‑pad *img* to exactly *slm_shape* (centre‑aligned)."""
    tgt_h, tgt_w = slm_shape
    h, w = img.shape
    out = np.zeros((tgt_h, tgt_w), dtype=img.dtype)
    y0, x0 = (tgt_h - h) // 2, (tgt_w - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img
    return out

###############################################################################
# Core class                                                                  #
###############################################################################

class OpticalJTCorrelator:
    """Real‑hardware Joint Transform Correlator.

    The class replicates the two‑pass binary/analogue JTC workflow from
    *binary_jtc_lightpipes.py*, but swaps out the numerical propagation
    for:
        • SLMdisplay.updateArray()   – writes patterns to an LCoS SLM
        • UC480Controller.snap()     – grabs frames from a camera at the
                                        Fourier plane.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        slm: SLMdisplay | None = None,
        cam: UC480Controller | None = None,
        *,
        slm_monitor: int = 1,
        cam_serial: str | None = None,
        binary_input: bool = True,
        binary_jps: bool = True,
        display_scale_factor: float = 0.05,
        sleep_time: float = 0.1,
        blocking_factor: float = 0.005,
    ) -> None:
        # Hardware handles ------------------------------------------------
        self.slm = slm or SLMdisplay(monitor=slm_monitor)
        self.cam = cam or UC480Controller(serial=cam_serial)

        # Operating parameters -------------------------------------------
        self.binary_input = binary_input
        self.binary_jps = binary_jps
        self.display_scale_factor = display_scale_factor
        self.sleep_time = sleep_time
        self.blocking_factor = blocking_factor

        # Cache SLM geometry ---------------------------------------------
        self.slm_w, self.slm_h = self.slm.getSize()
        print(
            f"[OpticalJTCorrelator] SLM: {self.slm_w}×{self.slm_h}  |  "
            f"binary_input={self.binary_input}  binary_jps={self.binary_jps}"
        )

    # ------------------------------------------------------------------
    # First pass – joint input ➜ JPS
    # ------------------------------------------------------------------

    def _upload_and_snap(self, img_8bit: np.ndarray) -> np.ndarray:
        """Upload *img_8bit* to the SLM and return a camera frame."""
        self.slm.updateArray(img_8bit)
        time.sleep(self.sleep_time)
        return self.cam.snap().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API – one full correlation cycle
    # ------------------------------------------------------------------

    def correlate(
        self,
        ref_digit: np.ndarray,
        obj_digit: np.ndarray,
        *,
        input_thresh=None,
        jps_thresh=None,
        return_planes: bool = False,
    ) -> Tuple[float, float, Tuple[int, int]] | Tuple[Any, ...]:
        """Run the two‑pass JTC and return correlation metrics.

        Returns (peak_val, central_dc, (dy, dx)) by default.  If
        *return_planes* is True, also returns the raw camera planes
        (JPS, correlation, masked correlation).
        """
        # --- 1. Joint input plane ---------------------------------------
        a0 = create_joint_input_plane(
            ref_digit,
            obj_digit,
            (self.slm_h, self.slm_w),
            input_thresh,
            display_scale_factor=self.display_scale_factor,
            binarize=self.binary_input,
        )

        if self.binary_input:
            a0_disp = ((a0 + 1) / 2 * 127).astype(np.uint8)
        else:
            a0_disp = np.round(a0 * 127).astype(np.uint8)

        jps_raw = self._upload_and_snap(a0_disp)

        # --- 2. JPS processing / second pass ----------------------------
        if self.binary_jps:
            thr = np.median(jps_raw) if jps_thresh is None else jps_thresh
            jps_phase = np.where(jps_raw > thr, 1.0, -1.0)
            jps_disp = ((jps_phase + 1) / 2 * 127).astype(np.uint8)
        else:
            jps_norm = jps_raw / (jps_raw.max() + EPS)
            jps_disp = np.round(jps_norm * 127).astype(np.uint8)

        jps_disp = _pad_to_slm(jps_disp, (self.slm_h, self.slm_w))
        corr_plane = self._upload_and_snap(jps_disp)

        # --- 3. Extract metrics -----------------------------------------
        cy, cx = np.array(corr_plane.shape) // 2
        central_dc = corr_plane.max()

        masked = corr_plane.copy()
        r = int(min(corr_plane.shape) * self.blocking_factor)
        masked[cy - r : cy + r + 1, cx - r : cx + r + 1] = 0.0

        peak_val = masked.max()
        py, px = np.unravel_index(masked.argmax(), masked.shape)
        dy, dx = py - cy, px - cx

        if return_planes:
            return (
                peak_val,
                central_dc,
                (dy, dx),
                jps_raw,
                corr_plane,
                masked,
            )

        return peak_val, central_dc, (dy, dx)

    # ------------------------------------------------------------------
    # Utility visualisation (optional)
    # ------------------------------------------------------------------

    @staticmethod
    def _show(img, ax, title, cmap="gray"):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        return im

    def debug_plot(
        self,
        joint_input_disp: np.ndarray,
        jps_raw: np.ndarray,
        corr_plane: np.ndarray,
        masked: np.ndarray,
    ) -> None:
        """Visual helper to verify each plane."""
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))
        self._show(joint_input_disp, axs[0], "Joint input (SLM)")
        self._show(jps_raw, axs[1], "JPS (camera)")
        self._show(corr_plane, axs[2], "Correlation plane")
        self._show(masked, axs[3], "Masked correlation")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Context‑manager sugar & tidy‑up
    # ------------------------------------------------------------------

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.slm.close()
        with contextlib.suppress(Exception):
            self.cam.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
