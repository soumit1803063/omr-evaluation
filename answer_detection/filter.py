def filter_overlapping_boxes(bbox_list, iou_threshold=0.5):
    def calculate_iou(box1, box2):
        x1, y1 = box1[0]
        x2, y2 = box1[1]
        x3, y3 = box2[0]
        x4, y4 = box2[1]

        # Calculate the intersection coordinates
        inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
        inter_x2, inter_y2 = min(x2, x4), min(y2, y4)

        # Calculate intersection area
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection_area = inter_width * inter_height

        # Calculate union area
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union_area = area1 + area2 - intersection_area

        # Avoid division by zero
        if union_area == 0:
            return 0
        return intersection_area / union_area

    #Sort boxes by confidence in descending order
    bbox_list = sorted(bbox_list, key=lambda x: x[1], reverse=True)

    #Filter boxes
    selected_boxes = []
    for current_box in bbox_list:
        is_overlapping = False
        for selected_box in selected_boxes:
            iou = calculate_iou(current_box[2:4], selected_box[2:4])
            if iou > iou_threshold:
                is_overlapping = True
                break
        if not is_overlapping:
            selected_boxes.append(current_box)

    return selected_boxes



        



