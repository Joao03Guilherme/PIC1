""" Abstract Camera controller
"""
 
from abc import ABC, abstractmethod
from typing import Any, Literal
from device_manager.camera.config.camera_config import (
    CameraConfig,
    default as default_config,
)
 
 
class Camera(ABC):
    """Abstract camera controller
 
    Args:
        config (CameraConfig): Camera config
    """
 
    @abstractmethod
    def __init__(self, config: CameraConfig | None = None):
        raise NotImplementedError
 
    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass  # The camera was not initialized
 
    @abstractmethod
    def snapshot(self) -> object:
        """Capture an image from the camera"""
        raise NotImplementedError
 
    @abstractmethod
    def close(self) -> None:
        """Disconect from the camera"""
        raise NotImplementedError
 
    @staticmethod
    def default_config() -> CameraConfig:
        """Returns a default config
 
        Returns:
            CameraConfig: Default configuration
        """
        return default_config()