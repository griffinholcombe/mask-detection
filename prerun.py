from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="face-mask-dataset/dataset.yaml",
    epochs=1,
    batch=1,
    workers=0,
    imgsz=640
)