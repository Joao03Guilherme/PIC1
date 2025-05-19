"""Thorlabs scientific camera controller
    See:
    * https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam
    * https://github.com/Thorlabs/Camera_Examples/tree/main/Python
"""
 
import numpy as np
import os
import sys
from typing import Optional
from device_manager.camera.camera import Camera
from device_manager.camera.config.camera_thorlabs_sci_config import (
    CameraThorlabsSciConfig,
    default as default_config,
)
 
 
try:
    if sys.platform in ["linux", "win32"]:
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        from instrumental.drivers.cameras import thorlabs  # type: ignore
 
        THORLABS_AVAILABLE = True
    else:
        THORLABS_AVAILABLE = False
except ImportError:
    THORLABS_AVAILABLE = False
    TLCameraSDK = None
    thorlabs = None
 
 
def get_ThorlabsCamera():
    if not THORLABS_AVAILABLE:
        raise ImportError(
            "Thorlabs TSI SDK is not available. "
            "This class cannot be used without installing the required SDK. This SDK is not currently available on MacOS."
        )
 
    class CameraThorlabsSci(Camera):
        """Thorlabs scientific camera controller"""
 
        @staticmethod
        def configure_path():
            """Configure the DLL path for Windows"""
            if sys.platform == "win32":
                # Get the path to the DLLs folder
                dll_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "dlls"
                )
                # Add the DLLs folder to the PATH
                os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]
 
        def __init__(self, config: CameraThorlabsSciConfig | None = None):
            try:
                # if on Windows, use the provided setup script to add the DLLs folder to the PATH
                CameraThorlabsSci.configure_path()
            except ImportError:
                pass
 
            # Start SDK
            self.sdk = TLCameraSDK()
 
            # Check if there are any cameras available
            available_cameras = self.sdk.discover_available_cameras()
 
            if len(available_cameras) < 1:
                raise IndexError("No thorcams detected")
 
            # Obtain config
            if config is None:
                self.config = CameraThorlabsSci.default_config()
            else:
                self.config = config
 
            camera_id = self.config["camera_id"]
 
            # Invalid/out of bounds instrumental camera id
            if len(available_cameras) - 1 < camera_id:
                raise IndexError("Invalid camera id")
 
            # Open the camera/device
            self.camera = self.sdk.open_camera(available_cameras[camera_id])
 
            # Apply extra configs to the camera
            if self.config["extra"] is not None:
                for key, value in self.config["extra"].items():
                    setattr(self.camera, key, value)
 
        def snapshot(self):
            """Capture a single frame from the camera
 
            Returns:
                numpy.ndarray: The captured image as a numpy array
            """
            self.camera.issue_software_trigger()
            frame = self.camera.get_pending_frame_or_null()
            if frame is None:
                raise RuntimeError("Failed to capture frame from Thorlabs camera")
            return frame.image_buffer
 
        def close(self):
            """Close the camera and release resources"""
            self.camera.dispose()
            self.sdk.dispose()
 
        @staticmethod
        def default_config() -> CameraThorlabsSciConfig:
            """Default configuration for Thorlabs camera
 
            Returns:
                CameraThorlabsSciConfig: Default configuration
            """
            return default_config()
 
    return CameraThorlabsSci