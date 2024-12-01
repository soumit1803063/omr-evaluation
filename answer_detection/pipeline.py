from .data_ingestion import data_ingestion
from .detection_model import DetectionModel
from .find_bboxes import get_ans
from .cluster import cluster_bboxes_by_y, get_slots, get_question_number
from .filter import filter_overlapping_boxes

def training_pipeline(base_model_path,trainned_model_path,data_path,epochs,image_size,device,patience):
    data_ingestion()  

    detector = DetectionModel(base_model_path)
    result = detector.train_model(
        data_path=data_path,
        epochs=epochs,
        img_size=image_size,
        device=device,
        patience=patience,
        trainned_model_path=trainned_model_path

    )
    return result, trainned_model_path


def prediction_pipeline(image, target_classes, model_path, total_rows=25, total_question=100):
    detector = DetectionModel(model_path)
    result = detector.predict(image)[0]
    bboxes = get_ans(result, target_classes)
    clustered_bboxes = cluster_bboxes_by_y(bboxes, num_clusters=total_rows)

    clusters_to_determine_slots = []
    for i in range(len(clustered_bboxes)):
        # Filter out overlapping boxes
        clustered_bboxes[i] = filter_overlapping_boxes(clustered_bboxes[i])

        # Sort each cluster by x-coordinate of the center point
        clustered_bboxes[i] = sorted(clustered_bboxes[i], key=lambda x: x[4][0])
        if len(clustered_bboxes[i]) == 4:
            clusters_to_determine_slots.append(clustered_bboxes[i])

    slots = get_slots(clusters_to_determine_slots)
    final_result = get_question_number(clustered_bboxes, total_question, total_rows, slots)
    return final_result
