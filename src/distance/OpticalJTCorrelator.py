from ..hardware.devices.Camera import UC480Controller
from ..hardware.devices.SLM import SLMdisplay
import numpy as np
import time
from PIL import Image  # Image manipulation utilities


class OpticalJTCorrelator:
    """
    Optical Joint Transform Correlator using a persistent hardware SLM and camera.

    All binary as well as analogue images handled by this class lie in the range 0–127.
    The class now guarantees that *both* the joint input plane and the processed JPS
    are zero-padded (and centred) to the full SLM resolution, so no deformation happens
    in the SLM driver.
    """

    # ------------------------------------------------------------------
    # Construction / setup
    # ------------------------------------------------------------------

    def __init__(
        self,
        slm: SLMdisplay = None,
        cam: UC480Controller = None,
        slm_monitor: int = 1,
        isImageLock: bool = False,
        alwaysTop: bool = False,
        cam_serial: str = None,
        sleep_time: float = 0.1,
        binary_input: bool = True,
        binary_jps: bool = True,
        display_scale_factor: float = 0.4,
    ):
        # operation modes
        self.binary_input = binary_input
        self.binary_jps = binary_jps
        self.display_scale_factor = display_scale_factor
        self.sleep_time = sleep_time

        # hardware
        self.slm = slm or SLMdisplay(
            monitor=slm_monitor,
            isImageLock=isImageLock,
            alwaysTop=alwaysTop,
        )
        self.cam = cam or UC480Controller(serial=cam_serial)

        # store SLM size once
        self.slm_width, self.slm_height = self.slm.getSize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _center_pad_to_slm(
        self, img: np.ndarray, pad_value: int = 0
    ) -> np.ndarray:
        """
        Return a copy of *img* that exactly matches the SLM resolution
        (self.slm_height × self.slm_width).  If the image is larger than the
        SLM, it is resized (bicubic) so that the longer side fits; otherwise it
        is simply inserted into a zero-filled canvas and centred.

        Parameters
        ----------
        img
            2-D uint8 array.
        pad_value
            Value to use for the padded background (default 0).

        Returns
        -------
        out : np.ndarray
            Padded (or resized) image, dtype uint8, shape (slm_height, slm_width).
        """
        h, w = img.shape
        slm_h, slm_w = self.slm_height, self.slm_width

        # resize down only if necessary
        if h > slm_h or w > slm_w:
            scale = min(slm_h / h, slm_w / w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = np.asarray(
                Image.fromarray(img, "L").resize((new_w, new_h), Image.BICUBIC),
                dtype=np.uint8,
            )
            h, w = img.shape

        canvas = np.full((slm_h, slm_w), pad_value, dtype=np.uint8)
        y0 = (slm_h - h) // 2
        x0 = (slm_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = img
        return canvas

    # ------------------------------------------------------------------
    #  Joint input-plane preparation
    # ------------------------------------------------------------------

    def create_joint_input_plane(
        self,
        digit_array_ref: np.ndarray,
        digit_array_obj: np.ndarray,
        thresh=None,
    ) -> np.ndarray:
        """
        Assemble the reference and object digits side-by-side, scale them
        (with an extra display_scale_factor < 1) and centre them on a black
        SLM canvas.  The returned image is uint8 and has the **exact SLM size**.

        If *binary_input* is True the final image is 0 / 127; otherwise it is
        analogue but still clipped to 0–127.
        """
        slm_rows, slm_cols = self.slm_height, self.slm_width

        # --- helper: bring any input array to uint8 0–255 ----------------
        def _prepare_digit(arr):
            if arr.max() <= 1.0 and arr.min() >= 0:
                return (arr * 255).astype(np.uint8)
            if arr.dtype != np.uint8:
                rng = arr.max() - arr.min() + 1e-6
                return ((arr - arr.min()) / rng * 255).astype(np.uint8)
            return arr

        ref_u8 = _prepare_digit(digit_array_ref)
        obj_u8 = _prepare_digit(digit_array_obj)

        combined = np.hstack((ref_u8, obj_u8))
        Hc, Wc = combined.shape
        img_comb = Image.fromarray(combined, "L")

        # --- scale to fill the SLM (but keep aspect) ---------------------
        scale_to_slm = min(slm_rows / Hc, slm_cols / Wc)
        new_w = int(Wc * scale_to_slm * self.display_scale_factor)
        new_h = int(Hc * scale_to_slm * self.display_scale_factor)
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        combined_resized = img_comb.resize((new_w, new_h), Image.BICUBIC)
        slm_canvas = np.zeros((slm_rows, slm_cols), dtype=np.float64)

        y0 = (slm_rows - new_h) // 2
        x0 = (slm_cols - new_w) // 2
        slm_canvas[y0 : y0 + new_h, x0 : x0 + new_w] = np.asarray(
            combined_resized, dtype=np.float64
        )

        # --- binarise / normalise to 0–127 -------------------------------
        if self.binary_input:
            t_val = (
                np.median(slm_canvas[slm_canvas > 0])
                if thresh is None and np.any(slm_canvas > 0)
                else (thresh if thresh is not None else 127)
            )
            binary = np.where(slm_canvas > t_val, 127, 0)
            return binary.astype(np.uint8)
        else:
            mn, mx = slm_canvas.min(), slm_canvas.max()
            if mx > mn:
                norm = (slm_canvas - mn) / (mx - mn) * 127.0
            else:
                norm = np.zeros_like(slm_canvas)
            return norm.astype(np.uint8)

    # ------------------------------------------------------------------
    #  Joint-power-spectrum processing
    # ------------------------------------------------------------------

    def process_jps(self, jps_image: np.ndarray, thresh=None) -> np.ndarray:
        """
        Convert the raw camera JPS into a displayable SLM image (uint8, 0–127)
        and **centre-pad** it to the SLM resolution.
        """
        # ---- binary branch ------------------------------------------------
        if self.binary_jps:
            t_val = np.median(jps_image) if thresh is None else thresh
            jps_proc = np.where(jps_image > t_val, 127, 0).astype(np.uint8)
            return self._center_pad_to_slm(jps_proc)

        # ---- analogue branch ---------------------------------------------
        if jps_image.dtype == np.uint8 and jps_image.max() <= 127:
            # already good
            return self._center_pad_to_slm(jps_image)

        mn, mx = jps_image.min(), jps_image.max()
        if mx > mn:
            norm = (jps_image.astype(np.float32) - mn) / (mx - mn) * 127.0
        else:
            norm = np.zeros_like(jps_image, dtype=np.float32)
        return self._center_pad_to_slm(norm.astype(np.uint8))

    # ------------------------------------------------------------------
    #  Main public API
    # ------------------------------------------------------------------

    def correlate(
        self,
        ref_image: np.ndarray,
        obj_image: np.ndarray,
        input_thresh=None,
        jps_thresh=None,
    ) -> np.ndarray:
        """
        Perform a full JTC cycle and return the correlation plane captured
        after the second camera exposure (no resizing/padding applied).
        """
        # 1. joint input plane
        joint_input = self.create_joint_input_plane(
            ref_image, obj_image, input_thresh
        )
        self.slm.updateArray(joint_input)
        time.sleep(self.sleep_time)

        # 2. capture JPS
        jps_raw = self.cam.snap()

        # 3. process + pad JPS, then write back
        jps_display = self.process_jps(jps_raw, jps_thresh)
        self.slm.updateArray(jps_display)
        time.sleep(self.sleep_time)

        # 4. final correlation read-out
        correlation_result = self.cam.snap()
        return correlation_result

    # ------------------------------------------------------------------
    #  Convenience setters
    # ------------------------------------------------------------------

    def set_binary_modes(self, *, binary_input=None, binary_jps=None):
        if binary_input is not None:
            self.binary_input = binary_input
        if binary_jps is not None:
            self.binary_jps = binary_jps

    def set_display_scale(self, scale_factor: float):
        self.display_scale_factor = max(0.01, min(1.0, scale_factor))

    # ------------------------------------------------------------------
    #  Camera helpers re-exposed
    # ------------------------------------------------------------------

    def set_exposure(self, ms: float):
        self.cam.set_exposure(ms)

    def set_roi(self, x: int, y: int, width: int, height: int, *, hbin=1, vbin=1):
        self.cam.set_roi(x, y, width, height, hbin=hbin, vbin=vbin)

    # ------------------------------------------------------------------
    #  Resource management
    # ------------------------------------------------------------------

    def close(self):
        with contextlib.suppress(Exception):
            self.slm.close()
        with contextlib.suppress(Exception):
            self.cam.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
