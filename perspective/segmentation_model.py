from ultralytics import YOLO

class DocumentSegmentationModel:
    def __init__(self, model_path='yolo11n-seg.pt'):
        self.model = YOLO(model_path)
    
    def train_model(self, data_path, epochs=100, img_size=640, device=0, patience=5,trainned_model_path=None):
        result = self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=img_size,
            device=device,
            patience=patience
        )
        if trainned_model_path:
            self.model.save(trainned_model_path)
        return result
    def predict(self, image):
        return self.model(image)