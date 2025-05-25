from ..hardware.devices.Camera import UC480Controller
from ..hardware.devices.SLM import SLMdisplay
import numpy as np
import time
from PIL import Image # Added for image manipulation

# Import the peak-and-shift helper from the computational JTC module
from .utils import _peak_and_shift


class OpticalJTCorrelator:
    """
    Optical Joint Transform Correlator using persistent hardware SLM and camera.

    Opens the SLM and camera once in the constructor and reuses them for all correlate() calls.
    Provides methods to configure exposure, ROI, and to cleanly close resources.
    Supports both binary and analog (grayscale) operation modes for input patterns and JPS processing.
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
        binary_input: bool = True, # New parameter
        binary_jps: bool = True,   # New parameter
        display_scale_factor: float = 0.4, # New parameter
    ):
        """
        Initialize the Optical JTC with binary/analog options.
        
        Args:
            slm: Pre-initialized SLMdisplay object.
            cam: Pre-initialized UC480Controller object.
            slm_monitor: Monitor index for the SLM.
            isImageLock: SLM image lock flag.
            alwaysTop: SLM window always on top flag.
            cam_serial: Camera serial number.
            sleep_time: Default wait time between SLM updates and camera captures.
            binary_input: If True, binarize input patterns to 0/255 values.
            binary_jps: If True, binarize the Joint Power Spectrum to 0/255.
            display_scale_factor: Scale factor for digit placement on SLM (0.0-1.0).
        """
        # Store operation modes
        self.binary_input = binary_input
        self.binary_jps = binary_jps
        self.display_scale_factor = display_scale_factor
        self.sleep_time = sleep_time # Renamed from self.sleep for clarity

        # Initialize or reuse SLM
        self.slm = slm or SLMdisplay(
            monitor=slm_monitor,
            isImageLock=isImageLock,
            alwaysTop=alwaysTop,
        )
        # Initialize or reuse camera
        self.cam = cam or UC480Controller(serial=cam_serial)
        
        # Query SLM resolution once
        self.slm_width, self.slm_height = self.slm.getSize() # Use more descriptive names

    def create_joint_input_plane(self, digit_array_ref: np.ndarray, digit_array_obj: np.ndarray, 
                                 thresh=None) -> np.ndarray:
        """
        Creates a joint input plane for the SLM hardware.
        The two input digit arrays (assumed to be 0-1 float or 0-255 uint8) 
        are placed side-by-side, then this combined image
        is scaled (preserving aspect ratio) to fit the SLM, further scaled by
        display_scale_factor, and centered on the SLM canvas.

        Args:
            digit_array_ref: 2D numpy array for the reference digit.
            digit_array_obj: 2D numpy array for the object/test digit.
            thresh: Binarization threshold (0-255) or None to use median. 
                    Only used if self.binary_input is True.
        Returns:
            A 2D numpy array (SLM shape, uint8) with values 0/255 if binary_input is True, 
            or 0-255 grayscale if binary_input is False.
        """
        slm_rows, slm_cols = self.slm_height, self.slm_width

        # Ensure input arrays are scaled to 0-255 uint8 for consistent processing
        def prepare_digit(digit_arr):
            if digit_arr.max() <= 1.0 and digit_arr.min() >= 0: # Likely 0-1 float
                return (digit_arr * 255.0).astype(np.uint8)
            elif digit_arr.dtype == np.uint8: # Already 0-255 uint8
                return digit_arr
            else: # Other cases, attempt to scale
                scaled = (digit_arr - digit_arr.min()) / (digit_arr.max() - digit_arr.min() + 1e-6) * 255.0
                return scaled.astype(np.uint8)

        ref_scaled_255 = prepare_digit(digit_array_ref)
        obj_scaled_255 = prepare_digit(digit_array_obj)

        # Combine digits side-by-side
        combined_digits_arr_raw = np.hstack((ref_scaled_255, obj_scaled_255))
        H_comb_raw, W_comb_raw = combined_digits_arr_raw.shape

        img_pil_combined = Image.fromarray(combined_digits_arr_raw, 'L')

        # Calculate dimensions to fit combined image onto SLM, preserving aspect ratio
        scale_h_slm = slm_rows / H_comb_raw
        scale_w_slm = slm_cols / W_comb_raw
        scale_slm = min(scale_h_slm, scale_w_slm)

        fit_pil_W_on_slm = int(W_comb_raw * scale_slm)
        fit_pil_H_on_slm = int(H_comb_raw * scale_slm)

        # Apply the display_scale_factor
        final_display_W = int(fit_pil_W_on_slm * self.display_scale_factor)
        final_display_H = int(fit_pil_H_on_slm * self.display_scale_factor)
        
        final_display_W = max(1, final_display_W) 
        final_display_H = max(1, final_display_H)

        img_resized_pil_combined = img_pil_combined.resize((final_display_W, final_display_H), Image.BICUBIC)
        
        slm_canvas_arr = np.zeros((slm_rows, slm_cols), dtype=np.float64) # Use float for processing
        
        paste_y = (slm_rows - final_display_H) // 2
        paste_x = (slm_cols - final_display_W) // 2
        
        resized_combined_arr = np.asarray(img_resized_pil_combined, dtype=np.float64)
        
        slm_canvas_arr[paste_y : paste_y + final_display_H, paste_x : paste_x + final_display_W] = resized_combined_arr
        
        if self.binary_input:
            t_val = np.median(slm_canvas_arr[slm_canvas_arr > 0]) if thresh is None and np.any(slm_canvas_arr > 0) else (thresh if thresh is not None else 127)
            # Ensure t_val is a scalar, not an array, if median of empty or all-zero is taken
            if not np.isscalar(t_val): t_val = 127 
            binary_result = np.where(slm_canvas_arr > t_val, 255.0, 0.0)
            return binary_result.astype(np.uint8)
        else:
            # Normalize to 0-255 range without binarizing
            min_val = slm_canvas_arr.min()
            max_val = slm_canvas_arr.max()
            if max_val > min_val:
                normalized = (slm_canvas_arr - min_val) / (max_val - min_val) * 255.0
                return normalized.astype(np.uint8)
            else: # Handle flat image (all same intensity)
                return np.full_like(slm_canvas_arr, int(min_val if min_val <=255 else 0), dtype=np.uint8)


    def process_jps(self, jps_image: np.ndarray, thresh=None) -> np.ndarray:
        """
        Process the Joint Power Spectrum according to self.binary_jps setting.
        Input jps_image is assumed to be a raw camera capture (typically uint8).
        
        Args:
            jps_image: Raw JPS image from camera.
            thresh: Binarization threshold (0-255) or None to use median.
                    Only used if self.binary_jps is True.
        Returns:
            Processed JPS (uint8) ready for display on SLM (0/255 if binary, 0-255 grayscale if analog).
        """
        if self.binary_jps:
            t_val = np.median(jps_image) if thresh is None else thresh
            binary_jps = np.where(jps_image > t_val, 255, 0) # Output 0 or 255
            return binary_jps.astype(np.uint8)
        else:
            # Normalize JPS to 0-255 for analog operation if not already
            if jps_image.dtype == np.uint8 and jps_image.min() == 0 and jps_image.max() == 255:
                return jps_image # Already in desired format
            
            min_val = jps_image.min()
            max_val = jps_image.max()
            if max_val > min_val:
                normalized = (jps_image.astype(np.float32) - min_val) / (max_val - min_val) * 255.0
                return normalized.astype(np.uint8)
            else: # Handle flat image
                 return np.full_like(jps_image, int(min_val if min_val <=255 else 0), dtype=np.uint8)

    def correlate(self, ref_image: np.ndarray, obj_image: np.ndarray, 
                                input_thresh=None, jps_thresh=None) -> np.ndarray:
        """
        Perform optical correlation using the hardware JTC with binary/analog options.
        
        Args:
            ref_image: Reference image (2D numpy array, e.g., 28x28).
            obj_image: Object image (2D numpy array, e.g., 28x28).
            input_thresh: Threshold for input binarization (if self.binary_input=True).
            jps_thresh: Threshold for JPS binarization (if self.binary_jps=True).
        Returns:
            Final correlation image (2D numpy array) from camera.
        """
        # Step 1: Create joint input plane
        joint_input = self.create_joint_input_plane(ref_image, obj_image, input_thresh)
        
        # Step 2: Display joint input on SLM
        self.slm.updateArray(joint_input)
        time.sleep(self.sleep_time)
        
        # Step 3: Capture Joint Power Spectrum with camera
        jps_raw = self.cam.snap() # Assuming cam.snap() returns a 2D numpy array
        
        # Step 4: Process JPS according to binary_jps setting
        jps_processed = self.process_jps(jps_raw, jps_thresh)
        
        # Step 5: Display processed JPS back on SLM
        self.slm.updateArray(jps_processed)
        time.sleep(self.sleep_time)
        
        # Step 6: Capture final correlation result
        correlation_result = self.cam.snap()
        
        return correlation_result

    def set_binary_modes(self, binary_input: bool = None, binary_jps: bool = None):
        """Update the binary operation modes."""
        if binary_input is not None:
            self.binary_input = binary_input
        if binary_jps is not None:
            self.binary_jps = binary_jps
            
    def set_display_scale(self, scale_factor: float):
        """Update the display scale factor for digits on SLM."""
        self.display_scale_factor = max(0.01, min(1.0, scale_factor)) # Clamp to reasonable range

    def set_exposure(self, ms: float) -> None: # Existing method, ensure it's kept
        """Set camera exposure time in milliseconds."""
        self.cam.set_exposure(ms)

    def set_roi( # Existing method, ensure it's kept
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

    def close(self) -> None: # Existing method
        """Close hardware resources cleanly."""
        try:
            self.slm.close()
        except Exception:
            pass # Or log error
        try:
            self.cam.close()
        except Exception:
            pass # Or log error

    def __enter__(self): # Existing method
        return self

    def __exit__(self, exc_type, exc, tb): # Existing method
        self.close()
