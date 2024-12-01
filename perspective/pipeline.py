from .data_ingestion import data_ingestion
from .segmentation_model import DocumentSegmentationModel
from .fing_bbox import find_largest_mask_and_bbox
from .adjust_corners import adjust_corners
from .transform_perspective import perspective_transform


def segmentation_training_pipeline(base_model_path,trainned_model_path,data_path,epochs,image_size,device,patience):
    data_ingestion()  

    detector = DocumentSegmentationModel(base_model_path)
    result = detector.train_model(
        data_path=data_path,
        epochs=epochs,
        img_size=image_size,
        device=device,
        patience=patience,
        trainned_model_path=trainned_model_path

    )
    return result,trainned_model_path



def perspective_transformation_pipeline(image,model_path):
    segmentor = DocumentSegmentationModel(model_path)

    segmentation_results = segmentor.predict(image)
    masks = segmentation_results[0].masks

    document_mask, document_area, document_bbox_corners = find_largest_mask_and_bbox(masks)

    if document_mask is not None:
        final_corners = adjust_corners(image, document_bbox_corners)
        cropped_image = perspective_transform(image, final_corners)
        return cropped_image
    else:
        print("No valid masks found.")

                                    
