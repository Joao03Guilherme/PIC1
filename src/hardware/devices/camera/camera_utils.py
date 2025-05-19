"""Utility functions for camera management"""
 
from typing import Dict, List
import subprocess
import re
import sys

from thor_labs_camera import (
    THORLABS_AVAILABLE,
    CameraThorlabsSci,
)

 
 
def _get_usb_cameras() -> List[Dict]:
    """Get list of USB cameras using system commands.
    
    Returns:
        List[Dict]: List of dictionaries containing camera information
    """
    cameras = []
    try:
        # On macOS, use system_profiler to list USB devices
        if sys.platform == "darwin":
            cmd = ["system_profiler", "SPUSBDataType"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # Look for camera devices in the output
            current_camera = None
            for line in output.split('\n'):
                if "Camera" in line or "Webcam" in line:
                    current_camera = {"name": line.strip()}
                elif current_camera and "Serial Number" in line:
                    current_camera["serial"] = line.split(":")[1].strip()
                    cameras.append(current_camera)
                    current_camera = None
                    
        # On Linux, use lsusb
        elif sys.platform == "linux":
            cmd = ["lsusb"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # Look for camera devices in the output
            for line in output.split('\n'):
                if "camera" in line.lower() or "webcam" in line.lower():
                    # Extract vendor and product IDs
                    match = re.search(r'ID (\w+):(\w+)', line)
                    if match:
                        vendor_id, product_id = match.groups()
                        cameras.append({
                            "name": line.split(":")[1].strip(),
                            "vendor_id": vendor_id,
                            "product_id": product_id
                        })
                        
        # On Windows, use Get-PnpDevice
        elif sys.platform == "win32":
            cmd = ["powershell", "Get-PnpDevice -Class Camera | Select-Object Status, Class, FriendlyName"]
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # Parse the output to find camera devices
            for line in output.split('\n'):
                if "Camera" in line:
                    cameras.append({
                        "name": line.split("Camera")[-1].strip(),
                        "status": "OK" if "OK" in line else "Unknown"
                    })
                    
    except Exception as e:
        print(f"Error listing USB cameras: {str(e)}")
        
    return cameras
 
 
def list_available_cameras() -> Dict[str, List[Dict]]:
    """List all available cameras connected to the system.
    
    Returns:
        Dict[str, List[Dict]]: Dictionary mapping camera types to lists of available devices.
            Each device is represented by a dictionary containing its properties (id, name, etc.).
            Example:
            {
                "opencv": [
                    {"id": 0, "name": "Integrated Camera"},
                    {"id": 1, "name": "USB Camera"}
                ],
                "thorlabs": [
                    {"id": 0, "name": "Thorlabs Camera 1", "serial": "12345"}
                ],
                "instrumental": [
                    {"id": 0, "name": "UC480 #1", "serial": "67890", "model": "Model X"}
                ]
            }
    """
    available_cameras = {}
    
    # List USB cameras first
    usb_cameras = _get_usb_cameras()
    if usb_cameras:
        available_cameras["usb"] = usb_cameras
    
    # List OpenCV cameras (only if we found USB cameras)
    if usb_cameras:
        try:
            import cv2
            opencv_cameras = []
            for i in range(len(usb_cameras)):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    name = f"Camera {i}"
                    try:
                        name = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                    except:
                        pass
                    opencv_cameras.append({"id": i, "name": name})
                    cap.release()
                
            if opencv_cameras:
                available_cameras["opencv"] = opencv_cameras
        except Exception as e:
            print(f"Error listing OpenCV cameras: {str(e)}")
 
    # List Thorlabs cameras
    if THORLABS_AVAILABLE and CameraThorlabsSci is not None:
        try:
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
            sdk = TLCameraSDK()
            cameras = sdk.discover_available_cameras()
            if cameras:
                thorlabs_cameras = []
                for i, cam in enumerate(cameras):
                    thorlabs_cameras.append({
                        "id": i,
                        "name": f"Thorlabs Camera {i}",
                        "serial": cam.serial_number,
                        "model": cam.model
                    })
                available_cameras["thorlabs"] = thorlabs_cameras
            sdk.dispose()
        except Exception as e:
            colorLog.debug(f"Error listing Thorlabs cameras: {str(e)}")
 

 
