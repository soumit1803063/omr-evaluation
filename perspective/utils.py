import cv2
import numpy as np

def read_image(image_path):
    return cv2.imread(image_path)

import cv2
import numpy as np

def resize_image(image, target_size=(640, 640)):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid image in the form of a NumPy array.")

    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    # Resize height first
    if target_height > original_height:
        interpolation_height = cv2.INTER_CUBIC  # Upscaling
    else:
        interpolation_height = cv2.INTER_AREA  # Downscaling

    resized_image = cv2.resize(image, (original_width, target_height), interpolation=interpolation_height)

    # Resize width next
    if target_width > resized_image.shape[1]:
        interpolation_width = cv2.INTER_CUBIC  # Upscaling
    else:
        interpolation_width = cv2.INTER_AREA  # Downscaling

    resized_image = cv2.resize(resized_image, (target_width, target_height), interpolation=interpolation_width)

    return resized_image






