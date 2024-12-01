import cv2
import numpy as np
def draw_corners(img, pts):
    for pt in pts:
        cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)  # Draw corners
    for i in range(len(pts)):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), (255, 0, 0), 2)  # Draw lines

def mouse_callback(event, x, y, flags, param):
    global selected_corner, document_bbox_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if mouse click is near any corner
        for i, corner in enumerate(document_bbox_corners):
            if np.linalg.norm([x - corner[0], y - corner[1]]) < 10:
                selected_corner = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and selected_corner != -1:
        # Drag selected corner
        document_bbox_corners[selected_corner] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_corner = -1


def adjust_corners(resized_image, initial_corners):
    global document_bbox_corners, selected_corner

    document_bbox_corners = list(initial_corners)  # Ensure corners are mutable
    selected_corner = -1

    # Create a window and set the mouse callback
    cv2.namedWindow("Adjust Corners")
    cv2.setMouseCallback("Adjust Corners", mouse_callback)

    # Interactive adjustment loop
    while True:
        img = resized_image.copy()
        draw_corners(img, document_bbox_corners)
        cv2.imshow("Adjust Corners", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 13:  # Enter to confirm
            print("Corners confirmed:", document_bbox_corners)
            break

    cv2.destroyAllWindows()

    # Return the final adjusted corners
    return document_bbox_corners