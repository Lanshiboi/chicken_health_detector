try:
    from flirpy.io.boson import Boson
    FLIRPY_AVAILABLE = True
except ImportError:
    FLIRPY_AVAILABLE = False

import cv2
import numpy as np

TEMP_MIN = 30.0
TEMP_MAX = 45.0

def extract_temperature_flir(image_path):
    """
    Extract temperature from a FLIR raw thermal image using flirpy if available,
    otherwise fallback to reading .bmp or other supported format image and estimating temperature from pixel intensities.
    Returns maximum temperature in Celsius.
    """
    if FLIRPY_AVAILABLE:
        try:
            with Boson(image_path) as camera:
                thermal_image = camera.grab()
            # Mask out low temperature pixels (background)
            mask = thermal_image > TEMP_MIN
            if np.any(mask):
                max_temp = np.percentile(thermal_image[mask], 90)
            else:
                max_temp = np.percentile(thermal_image, 90)
            return max_temp
        except Exception as e:
            raise RuntimeError(f"Failed to extract temperature from FLIR image: {e}")
    else:
        # Fallback: read image as grayscale and map pixel intensities to temperature range
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image file: {image_path}")
        norm = img.astype(np.float32) / 255.0
        temp_map = norm * (TEMP_MAX - TEMP_MIN) + TEMP_MIN
        # Mask out low temperature pixels (background)
        mask = temp_map > TEMP_MIN
        if np.any(mask):
            max_temp = np.percentile(temp_map[mask], 90)
        else:
            max_temp = np.percentile(temp_map, 90)
        return max_temp
