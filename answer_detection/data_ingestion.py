def data_ingestion():
    from roboflow import Roboflow
    rf = Roboflow(api_key="aObKsLqXs4LyTZdE9zjj")
    project = rf.workspace("sust-9qaee").project("omr-scanner")
    version = project.version(7)
    dataset = version.download("yolov11")
                
                
                