from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/mask_detector/weights/best.pt")

# Directly override the underlying PyTorch model names dictionary
model.model.names = {0: "No Mask", 1: "Mask"}

# Run prediction on validation images (since the test directory is missing)
results = model("face-mask-dataset/images/valid", save=True, conf=0.5)

# Print detection results
for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = box.conf.item()
        
        # This will now correctly pull "No Mask" or "Mask" from the overridden dictionary
        class_name = result.names[class_id]
        print(f"Detected: {class_name} ({confidence:.2f})")