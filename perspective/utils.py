import cv2
import numpy as np

def read_image(image_path):
    return cv2.imread(image_path)

def resize_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)





