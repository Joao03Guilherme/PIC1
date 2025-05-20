from ..hardware.devices.Camera import UC480Controller
from ..hardware.devices.SLM import SLMdisplay
import numpy as np
import time

# Import the peak-and-shift helper from the computational JTC module
from .utils import _peak_and_shift


class OpticalJTCorrelator:
    """
    Optical Joint Transform Correlator using persistent hardware SLM and camera.

    Opens the SLM and camera once in the constructor and reuses them for all correlate() calls.
    Provides methods to configure exposure, ROI, and to cleanly close resources.
    """

    def __init__(
        self,
        slm: SLMdisplay = None,
        cam: UC480Controller = None,
        slm_monitor: int = 1,
        isImageLock: bool = False,
        alwaysTop: bool = False,
        cam_serial: str = None,
        sleep_time: float = 0.1,
    ):
        # Initialize or reuse SLM
        self.slm = slm or SLMdisplay(
            monitor=slm_monitor,
            isImageLock=isImageLock,
            alwaysTop=alwaysTop,
        )
        # Initialize or reuse camera
        self.cam = cam or UC480Controller(serial=cam_serial)
        # Default wait time between updates
        self.sleep = sleep_time
        # Query SLM resolution once
        self.resX, self.resY = self.slm.getSize()

    def set_exposure(self, ms: float) -> None:
        """Set camera exposure time in milliseconds."""
        self.cam.set_exposure(ms)

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
        """Set region of interest on the camera sensor."""
        self.cam.set_roi(x, y, width, height, hbin=hbin, vbin=vbin)

    def correlate(
        self,
        img1_vec: np.ndarray,
        img2_vec: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[float, tuple[int, int], float, np.ndarray]:
        """
        Perform one optical JTC pass and return metrics.

        Parameters
        ----------
        img1_vec : array_like
            Flattened first image vector (values 0–255).
        img2_vec : array_like
            Flattened second image vector (values 0–255).
        shape : (H, W)
            Original image shape.

        Returns
        -------
        distance : float
            1 / normalized similarity.
        (dy, dx) : tuple[int, int]
            Pixel shift of the correlation peak.
        similarity : float
            Normalized correlation peak value.
        corr_plane_norm : np.ndarray
            Correlation plane normalized by 2||img1||·||img2||.
        """
        H, W = shape
        img1 = img1_vec.reshape(shape)
        img2 = img2_vec.reshape(shape)

        # Compute scaling to fit each half of the SLM
        half_w = self.resX // 2
        scale = min(half_w // W, self.resY // H)
        if scale < 1:
            raise ValueError(f"SLM {self.resX}×{self.resY} too small for image {W}×{H}")

        # Nearest-neighbor upsample and cast directly to uint8 (input already in 0–255 range)
        kron = lambda img: np.kron(img, np.ones((scale, scale), dtype=img.dtype))
        img1_up = kron(img1).astype(np.uint8)
        img2_up = kron(img2).astype(np.uint8)

        # Helper to normalize spectrum for display
        def to_uint8(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(np.float32)
            a = (a - a.min()) / (a.ptp() + 1e-12)
            return (255 * a).astype(np.uint8)

        # Create blank full frame buffer
        frame = np.zeros((self.resY, self.resX), dtype=np.uint8)
        dh, dw = img1_up.shape
        y0 = (self.resY - dh) // 2
        x_off = (half_w - dw) // 2

        # Place images side by side without additional normalization
        frame[y0 : y0 + dh, x_off : x_off + dw] = img1_up
        frame[y0 : y0 + dh, half_w + x_off : half_w + x_off + dw] = img2_up

        # First optical pass: display input and capture spectrum
        self.slm.updateArray(frame)
        time.sleep(self.sleep)
        spectrum = self.cam.snap()

        # Second optical pass: display spectrum and capture correlation
        spec_disp = to_uint8(spectrum)
        self.slm.updateArray(spec_disp)
        time.sleep(self.sleep)
        corr = self.cam.snap()

        # Analyze correlation using existing routine
        peak, (dy, dx) = _peak_and_shift(corr, shape)
        norm_val = np.linalg.norm(img1) * np.linalg.norm(img2)
        similarity = peak / (2 * norm_val) if norm_val else 0.0
        distance = 1.0 / similarity if similarity else np.inf
        corr_plane_norm = corr.astype(np.float32) / (2 * norm_val + 1e-12)

        return distance, (dy, dx), similarity, corr_plane_norm

    def close(self) -> None:
        """Close hardware resources cleanly."""
        try:
            self.slm.close()
        except Exception:
            pass
        try:
            self.cam.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
