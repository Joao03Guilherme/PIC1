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
            # Assuming bg_spectrum is already in a displayable range (e.g., captured camera data)
            bg_spec_disp = bg_spectrum.astype(np.uint8)
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

        # Prepare uint8 versions for display, assuming inputs are already 0-255 scaled
        img1_display = img1.astype(np.uint8, copy=False)
        img2_display = img2.astype(np.uint8, copy=False)

        # Create blank full frame buffer for SLM
        frame = np.zeros((self.resY, self.resX), dtype=np.uint8)
        # Define black frame for clearing SLM
        black_frame = np.zeros((self.resY, self.resX), dtype=np.uint8)

        # Calculate total width and height of the side-by-side images
        # H_img, W_img are from shape parameter (original image dimensions)
        combined_width = W * 2
        combined_height = H

        # Ensure the combined image fits on the SLM.
        if combined_width > self.resX or combined_height > self.resY:
            raise ValueError(
                f"Combined image size ({combined_width}x{combined_height}) "
                f"exceeds SLM resolution ({self.resX}x{self.resY}). "
                "Input images must be appropriately pre-sized."
            )

        # Calculate starting y position to center the block vertically
        y_start_slm = (self.resY - combined_height) // 2
        # Calculate starting x position to center the block horizontally
        x_start_slm_block = (self.resX - combined_width) // 2

        # Place img1_display into the frame
        frame[
            y_start_slm : y_start_slm + combined_height,
            x_start_slm_block : x_start_slm_block + W,
        ] = img1_display

        # Place img2_display into the frame, immediately to the right of img1_display
        frame[
            y_start_slm : y_start_slm + combined_height,
            x_start_slm_block + W : x_start_slm_block + combined_width,
        ] = img2_display

        # First optical pass: display input and capture spectrum
        self.slm.updateArray(frame)
        time.sleep(self.sleep)
        spectrum = self.cam.snap()
        # Clear SLM after displaying input
        self.slm.updateArray(black_frame)
        time.sleep(self.sleep)

        # Second optical pass: display spectrum as-is and capture correlation
        # Assuming spectrum is already in a displayable range (e.g., captured camera data)
        spec_disp = spectrum.astype(np.uint8)
        self.slm.updateArray(spec_disp)
        time.sleep(self.sleep)
        corr = self.cam.snap()
        # Clear SLM after displaying spectrum
        self.slm.updateArray(black_frame)
        time.sleep(self.sleep)

        # Subtract background bias if requested and available
        if subtract_bias and hasattr(self, "background_bias"):
            if corr.shape == self.background_bias.shape:
                corr = np.clip(corr - self.background_bias, 0, None)
            else:
                print(
                    f"Warning: Background shape {self.background_bias.shape} doesn't match correlation plane shape {corr.shape}. Skipping bias subtraction."
                )

        # Analyze correlation using existing routine
        peak, (dy, dx) = _peak_and_shift(corr, shape)
        norm_val = np.linalg.norm(img1) * np.linalg.norm(img2) / 1000
        similarity = peak / (2 * norm_val) if norm_val else 0.0
        distance = 1.0 / similarity if similarity else np.inf
        corr_plane_norm = corr.astype(np.float32) / (2 * norm_val + 1e-12)

        # Clear SLM by displaying a black screen
        black_frame = np.zeros((self.resY, self.resX), dtype=np.uint8)
        self.slm.updateArray(black_frame)
        time.sleep(self.sleep)  # Allow time for SLM to update

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
