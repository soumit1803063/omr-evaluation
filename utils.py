import os
import cv2
import numpy as np

def read_image(image_path):
    return cv2.imread(image_path)

def resize_image(image, target_size=(640, 640)):
    return cv2.resize(image, target_size)

def is_exist(file_path):
    return os.path.isfile(file_path)

def to_gray(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image_3D = cv2.merge([gray_image, gray_image, gray_image])
    return gray_image_3D

def create_directory(directory_path):
    if os.path.splitext(directory_path)[1]:
        os.makedirs(os.path.dirname(directory_path), exist_ok=True)
    else:
        os.makedirs(directory_path, exist_ok=True)


