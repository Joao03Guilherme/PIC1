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

        # Create the joined image block at original size using H, W from shape
        # H and W are the dimensions of a single original image
        joined_image_orig_height = H 
        joined_image_orig_width = W * 2
        
        # Ensure joined_image_block is only created and scaled if dimensions are valid
        if joined_image_orig_height > 0 and joined_image_orig_width > 0 and self.resY > 0 and self.resX > 0:
            joined_image_block = np.zeros((joined_image_orig_height, joined_image_orig_width), dtype=np.uint8)
            
            # Place img1 and img2 into the joined block
            joined_image_block[:, :W] = img1_display
            joined_image_block[:, W:W*2] = img2_display

            # Expand this joined block to the SLM's full resolution (self.resY, self.resX) 
            # into the existing 'frame' using nearest neighbor scaling.
            # 'frame' is already initialized as np.zeros((self.resY, self.resX), dtype=np.uint8).

            y_ratio = joined_image_orig_height / self.resY
            x_ratio = joined_image_orig_width / self.resX

            for y_slm in range(self.resY):
                for x_slm in range(self.resX):
                    # Find corresponding pixel in original joined image
                    y_orig_joined = int(y_slm * y_ratio)
                    x_orig_joined = int(x_slm * x_ratio)
                    
                    # Clamp coordinates to be within bounds of joined_image_block
                    y_orig_joined = min(y_orig_joined, joined_image_orig_height - 1)
                    x_orig_joined = min(x_orig_joined, joined_image_orig_width - 1)
                    
                    frame[y_slm, x_slm] = joined_image_block[y_orig_joined, x_orig_joined]
        else:
            # If original image dimensions are invalid or SLM resolution is zero,
            # 'frame' remains black (as initialized).
            print(f"Warning: Original image shape (H={H}, W={W}) or SLM resolution (resY={self.resY}, resX={self.resX}) is invalid for scaling. SLM frame will be black.")
            # 'frame' is already zeros, so no explicit action needed to make it black if it was already initialized to zeros.
            # If frame wasn't initialized to zeros, it should be explicitly set to black here.
            # Assuming 'frame' is correctly initialized to zeros before this block.

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

        # Center the spectrum on the SLM display
        spec_H, spec_W = spec_disp.shape
        slm_H, slm_W = self.resY, self.resX

        # Create a black frame the size of the SLM
        centered_spec_on_slm = np.zeros((slm_H, slm_W), dtype=np.uint8)

        # Calculate slices for copying the spectrum (cropping if necessary)
        # and for pasting onto the SLM frame (centering)

        # For Y dimension
        if spec_H <= slm_H:
            spec_y_start_crop = 0
            spec_y_end_crop = spec_H
            frame_y_start_paste = (slm_H - spec_H) // 2
            frame_y_end_paste = frame_y_start_paste + spec_H
        else: # spec_H > slm_H, crop spectrum
            spec_y_start_crop = (spec_H - slm_H) // 2
            spec_y_end_crop = spec_y_start_crop + slm_H
            frame_y_start_paste = 0
            frame_y_end_paste = slm_H

        # For X dimension
        if spec_W <= slm_W:
            spec_x_start_crop = 0
            spec_x_end_crop = spec_W
            frame_x_start_paste = (slm_W - spec_W) // 2
            frame_x_end_paste = frame_x_start_paste + spec_W
        else: # spec_W > slm_W, crop spectrum
            spec_x_start_crop = (spec_W - slm_W) // 2
            spec_x_end_crop = spec_x_start_crop + slm_W
            frame_x_start_paste = 0
            frame_x_end_paste = slm_W

        # Perform the copy
        centered_spec_on_slm[frame_y_start_paste:frame_y_end_paste, frame_x_start_paste:frame_x_end_paste] = \
            spec_disp[spec_y_start_crop:spec_y_end_crop, spec_x_start_crop:spec_x_end_crop]

        self.slm.updateArray(centered_spec_on_slm)
        time.sleep(self.sleep)
        corr = self.cam.snap()
        # Clear SLM after displaying spectrum

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
