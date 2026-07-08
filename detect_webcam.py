import cv2
import time
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/mask_detector/weights/best.pt")

# Define our own corrected class map locally
CLASS_NAMES = {
    0: "No Mask",
    1: "Mask"
}

# Define colors for each class (BGR format)
COLORS = {
    "Mask": (0, 255, 0),      # Green
    "No Mask": (0, 0, 255)    # Red
}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Face Mask Detector running. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Run detection on GPU
    results = model(frame, conf=0.5, verbose=False)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            
            # Translate class_id using our local dictionary instead of result.names
            class_name = CLASS_NAMES.get(class_id, "Unknown")

            # Get color based on class
            color = COLORS.get(class_name, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Face Mask Detector", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Detector stopped.")