import cv2
import numpy as np

# Function to apply perspective transform and crop the object based on corners
def perspective_transform(image, corners):
    # Define the source points (corners of the object)
    src_pts = np.array(corners, dtype=np.float32)

    # Define the destination points (desired rectangular shape, typically a square or a rectangle)
    # Here, we will define a rectangle of the same size as the input image
    width, height = image.shape[1], image.shape[0]
    dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transform to the image
    warped_image = cv2.warpPerspective(image, M, (width, height))

    return warped_image