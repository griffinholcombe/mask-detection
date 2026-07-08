import cv2
import os

# Path to training images and labels
images_dir = "face-mask-dataset/images/train"
labels_dir = "face-mask-dataset/labels/train"

# Class names and colors (green for Mask, red for No Mask)
class_names = ["No Mask", "Mask"]
class_colors = [(0, 0, 255), (0, 255, 0)]

# Get a sample image filename
image_files = os.listdir(images_dir)
sample_image = image_files[0]

# Load the image
img_path = os.path.join(images_dir, sample_image)
img = cv2.imread(img_path)
height, width = img.shape[:2]

# Load corresponding label file
label_file = os.path.splitext(sample_image)[0] + ".txt"
label_path = os.path.join(labels_dir, label_file)

# Draw bounding boxes from YOLO annotations
with open(label_path, "r") as f:
    for line in f.readlines():
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        # Convert from center format to corner format
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw rectangle and class label
        color = class_colors[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_names[class_id], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Display the annotated image
cv2.imshow("Sample Annotation", img)
print(f"Showing: {sample_image} with annotations from {label_file}")
print("Press any key to close the window.")
cv2.waitKey(0)
cv2.destroyAllWindows()