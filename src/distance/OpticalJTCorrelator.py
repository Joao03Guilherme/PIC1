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

    def calibrate(self, num_samples: int = 3) -> np.ndarray:
        """
        Measure the background bias in the optical system.
        
        This function displays black images and captures the resulting correlation plane,
        which represents the system's background bias. This bias can be subtracted
        from subsequent correlation measurements for more accurate results.
        
        Parameters
        ----------
        num_samples : int, default=3
            Number of background measurements to average.
            
        Returns
        -------
        bias : np.ndarray
            The average background correlation plane.
        """
        print("Calibrating optical JTC system...")
        
        # Create empty (black) image for input plane
        black_frame = np.zeros((self.resY, self.resX), dtype=np.uint8)
        
        # Accumulate multiple background readings
        background_corrs = []
        
        for i in range(num_samples):
            print(f"  Taking background sample {i+1}/{num_samples}")
            
            # First pass: display black input and capture spectrum
            self.slm.updateArray(black_frame)
            time.sleep(self.sleep)
            bg_spectrum = self.cam.snap()
            
            # Second pass: display spectrum and capture correlation
            bg_spec_disp = ((bg_spectrum - bg_spectrum.min()) / (bg_spectrum.ptp() + 1e-12) * 255).astype(np.uint8)
            self.slm.updateArray(bg_spec_disp) 
            time.sleep(self.sleep)
            bg_corr = self.cam.snap()
            
            background_corrs.append(bg_corr)
        
        # Average the backgrounds
        self.background_bias = np.mean(background_corrs, axis=0)
        print("Calibration complete.")
        
        return self.background_bias
        
    def correlate(
        self,
        img1_vec: np.ndarray,
        img2_vec: np.ndarray,
        shape: tuple[int, int],
        subtract_bias: bool = True,
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
        subtract_bias : bool, default=True
            Whether to subtract the calibrated background bias from the correlation plane.
            If True and calibration hasn't been performed, it will be done automatically.

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
        # If subtract_bias is True but no calibration has been done yet, do it now
        if subtract_bias and not hasattr(self, "background_bias"):
            self.calibrate()
            
        H, W = shape
        img1 = img1_vec.reshape(shape)
        img2 = img2_vec.reshape(shape)

        # Compute scaling to fit HALF of the available space in each SLM half
        half_w = self.resX // 2
        max_scale = min(half_w // W, self.resY // H)
        
        # Use only half of the maximum scale to make images smaller
        scale = max(max_scale // 2, 1)  # Ensure scale is at least 1
        
        # Nearest-neighbor upsample and cast directly to uint8 (input already in 0–255 range)
        kron = lambda img: np.kron(img, np.ones((scale, scale), dtype=img.dtype))
        
        # Scale images to 0-255 range if needed
        img1_scaled = ((img1 - img1.min()) / (img1.ptp() + 1e-12) * 255).astype(np.uint8)
        img2_scaled = ((img2 - img2.min()) / (img2.ptp() + 1e-12) * 255).astype(np.uint8)
        
        # Upsample the images (no binarization)
        img1_up = kron(img1_scaled).astype(np.uint8)
        img2_up = kron(img2_scaled).astype(np.uint8)

        # Create blank full frame buffer
        frame = np.zeros((self.resY, self.resX), dtype=np.uint8)
        dh, dw = img1_up.shape
        
        # Center the images vertically on the SLM
        y0 = (self.resY - dh) // 2
        
        # Center each image horizontally in its half of the SLM
        x_off1 = (half_w - dw) // 2
        x_off2 = half_w + (half_w - dw) // 2
        
        # Place images centered in each half of the SLM
        frame[y0 : y0 + dh, x_off1 : x_off1 + dw] = img1_up
        frame[y0 : y0 + dh, x_off2 : x_off2 + dw] = img2_up

        # First optical pass: display input and capture spectrum
        self.slm.updateArray(frame)
        time.sleep(self.sleep)
        spectrum = self.cam.snap()

        # Second optical pass: display spectrum as-is and capture correlation
        # Convert spectrum to 0-255 range for display
        spec_disp = ((spectrum - spectrum.min()) / (spectrum.ptp() + 1e-12) * 255).astype(np.uint8)
        self.slm.updateArray(spec_disp)
        time.sleep(self.sleep)
        corr = self.cam.snap()
        
        # Subtract background bias if requested and available
        if subtract_bias and hasattr(self, "background_bias"):
            if corr.shape == self.background_bias.shape:
                corr = np.clip(corr - self.background_bias, 0, None)
            else:
                print(f"Warning: Background shape {self.background_bias.shape} doesn't match correlation plane shape {corr.shape}. Skipping bias subtraction.")

        # Analyze correlation using existing routine
        peak, (dy, dx) = _peak_and_shift(corr, shape)
        norm_val = np.linalg.norm(img1) * np.linalg.norm(img2) / 1000
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
