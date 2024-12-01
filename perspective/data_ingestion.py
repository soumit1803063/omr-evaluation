def data_ingestion():
    from roboflow import Roboflow
    rf = Roboflow(api_key="aObKsLqXs4LyTZdE9zjj")
    project = rf.workspace("lung-x8el1").project("document-segmentation-v2-gt86h")
    version = project.version(2)
    dataset = version.download("yolov11")
                
                