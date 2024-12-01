import os

# Constants
TOTAL_QUESTIONS = 100
TOTAL_ROWS = 25
EPOCHS = 100
IMAGE_SIZE = 640
DEVICE = 0
PATIENCE = 5
CONFIDENCE_THRESHOLD = 0.20

TARGET_CLASSES = ['a', 'b', 'c', 'd', 'not answered']


# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Model and data configurations
MODEL_CONFIG = {
    "DETETION_BASE_MODEL": {
        "name": "yolo11n.pt",
        "dir": os.path.join(BASE_DIR, "model", "detection", "base_model")
    },
    "DETECTION_TRAINED_MODEL": {
        "name": "best.pt",
        "dir": os.path.join(BASE_DIR, "model", "detection","best_model")
    },
    "SEGMENTATION_BASE_MODEL": {
        "name": "yolo11n-seg.pt",
        "dir": os.path.join(BASE_DIR, "model", "segmentation", "base_model")
    },
    "SEGMENTATION_TRAINED_MODEL": {
        "name": "best.pt",
        "dir": os.path.join(BASE_DIR, "model", "segmentation","best_model")
    }
}

DATA_CONFIG = {
    "DETECTION": {
        "dir": os.path.join(BASE_DIR, "OMR-Scanner-7"),
        "data_yaml": "data.yaml"
    },
    "SEGMENTATION": {
        "dir": os.path.join(BASE_DIR, "document-segmentation-v2-2"),
        "data_yaml": "data.yaml"
    }
}

STATIC_CONFIG = {
    "ARTIFACT_DIR": os.path.join(BASE_DIR, 'static', 'artifacts'),
    "UPLOAD_DIR": os.path.join(BASE_DIR, 'static', 'uploads')
}


DETECTION_BASE_MODEL_PATH = os.path.join(MODEL_CONFIG["DETETION_BASE_MODEL"]["dir"], MODEL_CONFIG["DETETION_BASE_MODEL"]["name"])
DETECTION_TRAINED_MODEL_PATH = os.path.join(MODEL_CONFIG["DETECTION_TRAINED_MODEL"]["dir"], MODEL_CONFIG["DETECTION_TRAINED_MODEL"]["name"])
DETECTION_DATA_PATH = os.path.join(DATA_CONFIG["DETECTION"]["dir"], DATA_CONFIG["DETECTION"]["data_yaml"])

SEGMENTATION_BASE_MODEL_PATH = os.path.join(MODEL_CONFIG["SEGMENTATION_BASE_MODEL"]["dir"], MODEL_CONFIG["SEGMENTATION_BASE_MODEL"]["name"])
SEGMENTATION_TRAINED_MODEL_PATH = os.path.join(MODEL_CONFIG["SEGMENTATION_TRAINED_MODEL"]["dir"], MODEL_CONFIG["SEGMENTATION_TRAINED_MODEL"]["name"])
SEGMENTATION_DATA_PATH = os.path.join(DATA_CONFIG["SEGMENTATION"]["dir"], DATA_CONFIG["SEGMENTATION"]["data_yaml"])

ARTIFACT_DIR = STATIC_CONFIG["ARTIFACT_DIR"]
UPLOAD_DIR = STATIC_CONFIG["UPLOAD_DIR"]

# Exportable Configurations
__all__ = [
    "BASE_DIR", 

    "DETECTION_BASE_MODEL_PATH", 
    "DETECTION_TRAINED_MODEL_PATH", 
    "DETECTION_DATA_PATH", 
    
    "SEGMENTATION_BASE_MODEL_PATH", 
    "SEGMENTATION_TRAINED_MODEL_PATH", 
    "SEGMENTATION_DATA_PATH", 
    
    "ARTIFACT_DIR", 
    "UPLOAD_DIR", 
    
    "TOTAL_QUESTIONS", 
    "TOTAL_ROWS", 
    "EPOCHS", 
    "IMAGE_SIZE", 
    "DEVICE", 
    "PATIENCE", 
    "CONFIDENCE_THRESHOLD",
    "TARGET_CLASSES"
]
