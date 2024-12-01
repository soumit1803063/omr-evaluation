import cv2
import numpy as np
def to_binary_mask(mask_tensor):
    mask_cpu = mask_tensor.data.permute(1, 2, 0).cpu().numpy()
    return (mask_cpu > 0).astype(np.uint8)

def calculate_mask_area(mask):
    return (mask > 0).sum()

def find_bounding_box(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        #Four corners of bounding box
        return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    return None

def find_largest_mask_and_bbox(masks):
    largest_mask = None
    largest_area = 0
    bounding_box = None

    for mask in masks:
        binary_mask = to_binary_mask(mask)
        mask_area = calculate_mask_area(binary_mask)
        
        if mask_area > largest_area:
            largest_area = mask_area
            largest_mask = binary_mask
            bounding_box = find_bounding_box(binary_mask)
    
    return largest_mask, largest_area, bounding_box