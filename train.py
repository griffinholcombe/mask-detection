from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="face-mask-dataset/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        name="mask_detector"
    )

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")