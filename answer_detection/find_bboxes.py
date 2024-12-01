def get_ans(result,target_classes,threshold_conf=0.2):

    boxes = result.boxes
    class_names = result.names
    
    ans_boxes = []
    for box in boxes:
        cls_id = int(box.cls)  # Class ID of the detected object
        confidence = float(box.conf)  # Confidence score of the detection
        class_name = class_names[cls_id]
        if class_name in target_classes and confidence > threshold_conf:
            bbox = box.xyxy.cpu().numpy()[0]
            top_left = (int(bbox[0]),int(bbox[1]))
            bottom_right = (int(bbox[2]),int(bbox[3]))
            center_point = (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))
            bbox = (class_name,confidence,top_left,bottom_right,center_point)
            ans_boxes.append(bbox)

    return ans_boxes
