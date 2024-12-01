import os
import uuid
import cv2
from flask import Flask, render_template, request, send_from_directory, url_for

from perspective.pipeline import (
    perspective_transformation_pipeline,
    segmentation_training_pipeline
)
from answer_detection.pipeline import training_pipeline, prediction_pipeline
from utils import (
    create_directory,
    is_exist,
    read_image,
    resize_image,
    to_gray
)
from config import (
    UPLOAD_DIR, ARTIFACT_DIR, SEGMENTATION_BASE_MODEL_PATH, SEGMENTATION_TRAINED_MODEL_PATH,
    DETECTION_BASE_MODEL_PATH, DETECTION_TRAINED_MODEL_PATH, SEGMENTATION_DATA_PATH,
    DETECTION_DATA_PATH, EPOCHS, IMAGE_SIZE, DEVICE, PATIENCE, TARGET_CLASSES,
    TOTAL_ROWS, TOTAL_QUESTIONS
)

# Flask App Initialization
app = Flask(__name__)
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['ARTIFACT_DIR'] = ARTIFACT_DIR

# Ensure directories exist
for directory in [
    UPLOAD_DIR, ARTIFACT_DIR, SEGMENTATION_BASE_MODEL_PATH, SEGMENTATION_TRAINED_MODEL_PATH,
    DETECTION_BASE_MODEL_PATH, DETECTION_TRAINED_MODEL_PATH
]:
    create_directory(directory)

# Color mapping for detected answers
COLOR_MAP = {
    'a': (255, 0, 0),         # Blue
    'b': (0, 255, 0),         # Green
    'c': (0, 50, 255),    
    'd': (255, 255, 0),       # Cyan
    'not answered': (255, 0, 255)  # Magenta
}

# Utility Functions
def ensure_model_trained(base_model_path, trained_model_path, data_path,option):
    """Ensure the required model is trained."""
    if not is_exist(trained_model_path):
        if option == 0:
            segmentation_training_pipeline(
                base_model_path=base_model_path,
                trainned_model_path=trained_model_path,
                data_path=data_path,
                epochs=EPOCHS,
                image_size=IMAGE_SIZE,
                device=DEVICE,
                patience=PATIENCE
            )
        else:
            training_pipeline(
                base_model_path=base_model_path,
                trainned_model_path=trained_model_path,
                data_path=data_path,
                epochs=EPOCHS,
                image_size=IMAGE_SIZE,
                device=DEVICE,
                patience=PATIENCE
            )


def annotate_image(image, result):
    """Annotate the detected answers on the image."""
    for key, value in result.items():
        if value:
            tl, br, op = value['tl'], value['br'], value['op']
            color = COLOR_MAP.get(op, (255, 255, 255))  # Default color: white
            cv2.rectangle(image, tl, br, color, 2)

            text = op[0].upper()  # Shortened label
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            text_x = br[0] + 5
            text_y = tl[1] + (br[1] - tl[1]) // 2 + text_height // 2
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image

# Routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image uploads and process them."""
    file = request.files.get('image')
    if not file or file.filename == '':
        return "No file selected", 400

    # Save uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    random_filename = f"{uuid.uuid4().hex}{file_extension}"
    uploaded_image_path = os.path.join(app.config['UPLOAD_DIR'], random_filename)
    file.save(uploaded_image_path)

    # Ensure models are trained
    ensure_model_trained(
        SEGMENTATION_BASE_MODEL_PATH, SEGMENTATION_TRAINED_MODEL_PATH, SEGMENTATION_DATA_PATH,0
    )
    ensure_model_trained(
        DETECTION_BASE_MODEL_PATH, DETECTION_TRAINED_MODEL_PATH, DETECTION_DATA_PATH,1
    )
    # tp = "/home/somu/Documents/OMR_Evaluation/result/For Review/Data-2"

    # Process image
    uploaded_image = read_image(uploaded_image_path)
    # cv2.imwrite(os.path.join(tp,"input image.png"), uploaded_image)
    resized_image = resize_image(uploaded_image)
    # cv2.imwrite(os.path.join(tp,"resized image.png"), resized_image)
    transformed_image = perspective_transformation_pipeline(resized_image,
                                                            SEGMENTATION_TRAINED_MODEL_PATH)
    # cv2.imwrite(os.path.join(tp,"perspective transformed image.png"), transformed_image)
    transformed_gray_image = to_gray(transformed_image)

    # Predict answers
    result = prediction_pipeline(
        image=transformed_gray_image,
        target_classes=TARGET_CLASSES,
        model_path=DETECTION_TRAINED_MODEL_PATH,
        total_rows=TOTAL_ROWS,
        total_question=TOTAL_QUESTIONS
    )

    # Annotate the image
    annotated_image = annotate_image(transformed_image, result)
    # cv2.imwrite(os.path.join(tp,"detection image.png"), annotated_image)

    # Save the transformed and annotated image
    transformed_image_filename = f"{uuid.uuid4().hex}.jpg"
    transformed_image_path = os.path.join(app.config['ARTIFACT_DIR'], transformed_image_filename)
    cv2.imwrite(transformed_image_path, annotated_image)

    # Generate URL for the transformed image
    transformed_image_url = url_for('uploaded_transformed_file', filename=transformed_image_filename)

    # Prepare result data for rendering
    result_data = [{'key': key, 'op': value['op'] if value else "Not Detected"} for key, value in result.items()]

    return render_template('result.html', image_url=transformed_image_url, result_data=result_data)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_DIR'], filename)

@app.route('/artifacts/<filename>')
def uploaded_transformed_file(filename):
    """Serve transformed files."""
    return send_from_directory(app.config['ARTIFACT_DIR'], filename)

# Entry Point
if __name__ == '__main__':
    app.run(debug=True)
