from __future__ import annotations
import contextlib
from ..hardware.devices.Camera import UC480Controller
from ..hardware.devices.SLM import SLMdisplay
import numpy as np
import time
from PIL import Image  # Image manipulation utilities
import matplotlib.pyplot as plt

# Small value to avoid division by zero
EPS = 1e-6


class OpticalJTCorrelator:
    """
    Optical Joint Transform Correlator using a persistent hardware SLM and camera.

    This implementation matches the functionality of binary_jtc_lightpipes.py but uses
    real hardware components instead of simulation.

    All binary as well as analogue images handled by this class lie in the range 0–127.
    The class guarantees that both the joint input plane and the processed JPS
    are zero-padded (and centered) to the full SLM resolution, preventing distortion.
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
        display_scale_factor: float = 0.05,  # Match default in binary_jtc_lightpipes.py
        blocking_factor: float = 0.005,  # Match default for DC blocking in binary_jtc_lightpipes.py
    ):
        # operation modes - match binary_jtc_lightpipes.py defaults
        self.binary_input = binary_input
        self.binary_jps = binary_jps
        self.display_scale_factor = display_scale_factor
        self.sleep_time = sleep_time
        self.blocking_factor = blocking_factor

        # hardware
        self.slm = slm or SLMdisplay(
            monitor=slm_monitor,
            isImageLock=isImageLock,
            alwaysTop=alwaysTop,
        )
        self.cam = cam or UC480Controller(serial=cam_serial)

        # store SLM size once - this corresponds to the simulation grid size N×N
        self.slm_width, self.slm_height = self.slm.getSize()

        # Print configuration information
        print(f"OpticalJTCorrelator initialized with:")
        print(f"  SLM resolution: {self.slm_width}×{self.slm_height}")
        print(f"  Binary input: {self.binary_input}, Binary JPS: {self.binary_jps}")
        print(f"  Display scale factor: {self.display_scale_factor}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _center_pad_to_slm(
        self, img: np.ndarray, pad_value: int = 0
    ) -> np.ndarray:
        """
        Return a copy of *img* that exactly matches the SLM resolution.
        If the image is larger than the SLM, it is resized (bicubic) so that
        the longer side fits; otherwise it is inserted into a zero-filled canvas and centered.

        This matches make_slm_aperture from binary_jtc_lightpipes.py.

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
                Image.fromarray(img, mode="L").resize((new_w, new_h), Image.BICUBIC),
                dtype=np.uint8,
            )
            h, w = img.shape

        canvas = np.full((slm_h, slm_w), pad_value, dtype=np.uint8)
        y0 = (slm_h - h) // 2
        x0 = (slm_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = img
        return canvas

    # ------------------------------------------------------------------
    #  Joint input-plane preparation (matching binary_jtc_lightpipes.py)
    # ------------------------------------------------------------------

    def create_joint_input_plane(
        self,
        digit_array_ref: np.ndarray,
        digit_array_obj: np.ndarray,
        thresh=None,
    ) -> np.ndarray:
        """
        Assemble the reference and object digits side-by-side, scale them
        (with display_scale_factor) and center them on a black SLM canvas.

        This matches create_joint_input_plane from binary_jtc_lightpipes.py.

        Parameters
        ----------
        digit_array_ref, digit_array_obj
            The reference and object patterns to be placed side-by-side
        thresh
            Threshold value for binarization. If None, uses median.

        Returns
        -------
        np.ndarray
            SLM-ready image with reference and object patterns side-by-side
        """
        slm_rows, slm_cols = self.slm_height, self.slm_width

        # --- Scale input arrays to 0-255 range (uint8) -------------------
        def scale_digit_to_255(d):
            m = d.max()
            return np.zeros_like(d, np.uint8) if m == 0 else (d / m * 255).astype(np.uint8)

        ref_255 = scale_digit_to_255(digit_array_ref)
        obj_255 = scale_digit_to_255(digit_array_obj)

        # --- Combine images side-by-side ---------------------------------
        combo = np.hstack((ref_255, obj_255))
        H0, W0 = combo.shape

        # --- Calculate scale to fit within SLM while maintaining aspect ratio ---
        scale = min(slm_rows / H0, slm_cols / W0)
        Wf = int(W0 * scale * self.display_scale_factor)
        Hf = int(H0 * scale * self.display_scale_factor)
        Wf, Hf = max(1, Wf), max(1, Hf)

        # --- Resize and center on canvas ---------------------------------
        combo_rs = Image.fromarray(combo).resize((Wf, Hf), Image.BICUBIC)
        canvas = np.zeros((slm_rows, slm_cols), float)
        y0, x0 = (slm_rows - Hf) // 2, (slm_cols - Wf) // 2
        canvas[y0 : y0 + Hf, x0 : x0 + Wf] = np.asarray(combo_rs, float)

        # --- Binarize or normalize ---------------------------------------
        if self.binary_input:
            t = np.median(canvas) if thresh is None else thresh
            # Match binary_jtc_lightpipes.py: use 0.0 and 1.0 (will be rescaled to 0-127 for SLM)
            return np.where(canvas > t, 1.0, -1.0)

        # For analog mode, normalize 0-1 (will be rescaled to 0-127 for SLM)
        return (canvas - canvas.min()) / (canvas.ptp() + EPS)

    # ------------------------------------------------------------------
    #  Joint-power-spectrum processing (matching binary_jtc_lightpipes.py)
    # ------------------------------------------------------------------

    def process_jps(self, jps_image: np.ndarray, thresh=None) -> np.ndarray:
        """
        Process the JPS based on binary_jps flag, matching the logic in
        perform_jtc_correlation from binary_jtc_lightpipes.py.

        Parameters
        ----------
        jps_image : np.ndarray
            Raw JPS from camera
        thresh : float, optional
            Threshold for binarization. If None, uses median.

        Returns
        -------
        np.ndarray
            Processed JPS ready for SLM display (0-127 uint8)
        """
        # Ensure we're working with float for processing
        jps_float = jps_image.astype(np.float32)

        # Process according to binarization preference
        if self.binary_jps:
            # Binary JPS processing (matching binary_jtc_lightpipes.py)
            thr_JPS = np.median(jps_float) if thresh is None else thresh
            JPS_bin = np.where(jps_float > thr_JPS, 1.0, -1.0)
            # Rescale from -1.0/1.0 to 0-127 for SLM
            processed = ((JPS_bin + 1) / 2 * 127).astype(np.uint8)
        else:
            # Analog JPS processing (matching binary_jtc_lightpipes.py)
            JPS_norm = jps_float / jps_float.max() if jps_float.max() > 0 else jps_float
            # Rescale from 0-1.0 to 0-127 for SLM
            processed = (JPS_norm * 127).astype(np.uint8)

        # Ensure the JPS is centered and padded to match SLM dimensions
        return self._center_pad_to_slm(processed)

    # ------------------------------------------------------------------
    #  Main correlation engine (matching perform_jtc_correlation)
    # ------------------------------------------------------------------

    def correlate(
        self,
        ref_image: np.ndarray,
        obj_image: np.ndarray,
        input_thresh=None,
        jps_thresh=None,
        return_debug_info=False,
    ) -> tuple:
        """
        Perform a full JTC cycle matching the behavior of perform_jtc_correlation
        in binary_jtc_lightpipes.py.

        Parameters
        ----------
        ref_image, obj_image
            The reference and object images to correlate
        input_thresh, jps_thresh
            Optional thresholds for input and JPS binarization
        return_debug_info
            If True, returns additional debug information

        Returns
        -------
        Tuple of (peak_val, central_dc, (dy, dx), correlation_plane)
            - peak_val: Value of highest correlation peak after DC blocking
            - central_dc: Value of central DC peak
            - (dy, dx): Position of highest correlation peak relative to center
            - correlation_plane: Full correlation output intensity from camera
        """
        # 1. Create joint input plane (like binary_jtc_lightpipes.py)
        joint_input = self.create_joint_input_plane(
            ref_image, obj_image, input_thresh
        )

        # Convert joint input for SLM display (rescale from -1/1 or 0-1 to 0-127)
        if self.binary_input:
            joint_input_display = ((joint_input + 1) / 2 * 127).astype(np.uint8)
        else:
            joint_input_display = (joint_input * 127).astype(np.uint8)

        # Send to SLM
        self.slm.updateArray(joint_input_display)
        time.sleep(self.sleep_time)

        # 2. Capture JPS with camera
        jps_raw = self.cam.snap()

        # 3. Process JPS and display on SLM
        jps_display = self.process_jps(jps_raw, jps_thresh)
        self.slm.updateArray(jps_display)
        time.sleep(self.sleep_time)

        # 4. Capture final correlation plane
        correlation_plane = self.cam.snap()

        # 5. Process correlation plane to match binary_jtc_lightpipes.py output
        # Extract peak values and locations
        center_y, center_x = correlation_plane.shape[0] // 2, correlation_plane.shape[1] // 2
        central_dc = correlation_plane.max()  # DC term is typically the brightest

        # Block central region to find correlation peak (like binary_jtc_lightpipes.py)
        corr_masked = correlation_plane.copy()
        half_block = int(min(correlation_plane.shape) * self.blocking_factor)
        corr_masked[
            center_y - half_block : center_y + half_block + 1,
            center_x - half_block : center_x + half_block + 1,
        ] = 0.0

        # Find peak and its location
        peak_val = corr_masked.max()
        peak_y, peak_x = np.unravel_index(np.argmax(corr_masked), corr_masked.shape)
        dy, dx = peak_y - center_y, peak_x - center_x

        if return_debug_info:
            return (
                peak_val,
                central_dc,
                (dy, dx),
                correlation_plane,
                joint_input_display,
                jps_display,
                corr_masked,
            )
        return peak_val, central_dc, (dy, dx), correlation_plane

    # ------------------------------------------------------------------
    # Plotting/visualization functions (matching binary_jtc_lightpipes.py)
    # ------------------------------------------------------------------

    def plot_results(
        self, joint_input, correlation_plane, corr_masked=None, zoom_factor=4
    ):
        """
        Plot the joint input plane and correlation results, matching the
        visualization in binary_jtc_lightpipes.py.

        Parameters
        ----------
        joint_input : np.ndarray
            The joint input plane image
        correlation_plane : np.ndarray
            The full correlation result
        corr_masked : np.ndarray, optional
            Correlation plane with DC term masked
        zoom_factor : int
            Factor to zoom in on correlation plane center
        """
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # Plot joint input plane
        if self.binary_input:
            # Convert from 0-127 back to 0-1 for display
            a0_display = joint_input / 127.0
        else:
            a0_display = joint_input / 127.0

        im_a0 = axs[0].imshow(a0_display, cmap="gray")
        axs[0].set_title("Joint Input Plane")
        axs[0].set_xlabel("X Position (pixels)")
        axs[0].set_ylabel("Y Position (pixels)")
        fig.colorbar(im_a0, ax=axs[0], label="Amplitude")

        # Plot correlation plane (zoomed)
        center_y, center_x = correlation_plane.shape[0] // 2, correlation_plane.shape[1] // 2
        zoom_size = min(center_y, center_x) // zoom_factor

        zoom_slice_y = slice(center_y - zoom_size, center_y + zoom_size)
        zoom_slice_x = slice(center_x - zoom_size, center_x + zoom_size)

        corr_to_show = correlation_plane if corr_masked is None else corr_masked

        im_corr = axs[1].imshow(
            corr_to_show[zoom_slice_y, zoom_slice_x],
            cmap="hot",
            extent=[
                center_x - zoom_size,
                center_x + zoom_size,
                center_y + zoom_size,
                center_y - zoom_size,
            ],
        )
        axs[1].set_title(f"Correlation Plane (Zoomed {zoom_factor}×)")
        axs[1].set_xlabel("X Position (pixels from center)")
        axs[1].set_ylabel("Y Position (pixels from center)")
        fig.colorbar(im_corr, ax=axs[1], label="Intensity")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Added utility functions for testing all digits (like binary_jtc_lightpipes.py)
    # ------------------------------------------------------------------

    def compare_digits(
        self, ref_digit, test_digits, dataset_images, dataset_labels
    ):
        """
        Compare a reference digit against multiple test digits, similar to
        the functionality added to binary_jtc_lightpipes.py.

        Parameters
        ----------
        ref_digit : int
            Label of the reference digit
        test_digits : list
            List of digit labels to test against
        dataset_images : np.ndarray
            Images from the dataset
        dataset_labels : np.ndarray
            Labels for the dataset images

        Returns
        -------
        dict
            Dictionary of results with digit labels as keys
        """
        # Get the reference digit image
        ref_indices = np.where(dataset_labels == ref_digit)[0]
        if len(ref_indices) == 0:
            raise ValueError(f"Reference digit {ref_digit} not found in dataset")
        ref_img = dataset_images[ref_indices[0]].reshape(28, 28) / 255.0

        results = {}
        print(f"Reference Digit: {ref_digit}")
        print("Comparing against specified test digits...")

        for test_digit in test_digits:
            print(f"  Testing {ref_digit} vs {test_digit}...")

            test_indices = np.where(dataset_labels == test_digit)[0]
            if len(test_indices) == 0:
                print(f"  Test digit {test_digit} not found in dataset, skipping")
                continue

            test_img = dataset_images[test_indices[0]].reshape(28, 28) / 255.0

            # Run correlation
            peak_val, central_dc, peak_coords, _ = self.correlate(ref_img, test_img)

            # Calculate normalized peak
            norm_peak = peak_val / (central_dc + EPS)

            results[test_digit] = {
                "peak_val": peak_val,
                "central_dc": central_dc,
                "norm_peak": norm_peak,
                "peak_coords": peak_coords,
            }

            print(f"    Normalized peak for {ref_digit} vs {test_digit}: {norm_peak:.4f}")

        return results

    def plot_comparison_results(self, results):
        """
        Plot the results from compare_digits as a bar chart.

        Parameters
        ----------
        results : dict
            Results from compare_digits function
        """
        digits = list(results.keys())
        norm_peaks = [results[d]["norm_peak"] for d in digits]

        plt.figure(figsize=(10, 7))
        plt.bar(digits, norm_peaks, color="skyblue")
        plt.xlabel("Test Digit")
        plt.ylabel("Normalized Correlation Peak (Peak/DC)")
        plt.title(f"Correlation Strength vs. Test Digits")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    #  Convenience setters
    # ------------------------------------------------------------------

    def set_binary_modes(self, *, binary_input=None, binary_jps=None):
        """Set binary modes for input and JPS processing."""
        if binary_input is not None:
            self.binary_input = binary_input
        if binary_jps is not None:
            self.binary_jps = binary_jps

    def set_display_scale(self, scale_factor: float):
        """Set display scale factor for the input images."""
        self.display_scale_factor = max(0.01, min(1.0, scale_factor))

    def set_blocking_factor(self, factor: float):
        """Set the blocking factor for DC term masking."""
        self.blocking_factor = max(0.001, min(0.5, factor))

    # ------------------------------------------------------------------
    #  Camera helpers re-exposed
    # ------------------------------------------------------------------

    def set_exposure(self, ms: float):
        """Set camera exposure time in milliseconds."""
        self.cam.set_exposure(ms)

    def set_roi(self, x: int, y: int, width: int, height: int, *, hbin=1, vbin=1):
        """Set camera region of interest and binning."""
        self.cam.set_roi(x, y, width, height, hbin=hbin, vbin=vbin)

    # ------------------------------------------------------------------
    #  Resource management
    # ------------------------------------------------------------------

    def close(self):
        """Close hardware resources."""
        try:
            if self.slm:
                self.slm.close()
        except Exception as e:
            print(f"Error closing SLM: {e}")

        try:
            if self.cam:
                self.cam.close()
        except Exception as e:
            print(f"Error closing camera: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
