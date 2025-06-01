from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')
#yolov8m is better than yolov8n for small object detection
# Perform inference on the input image
image_path = 'test5.jpg'
#results = model(image_path, conf=0.29, iou=0.001, imgsz=1280) #imgsz must be multiple of 32.
results = model(image_path)
# Open a text file to save the results, and empty it at the start
with open('detection_results.txt', 'w') as f:
    # Iterate over each detected object
    for box in results[0].boxes:
        # Extract class ID and bounding box coordinates
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the class name from the model's class names
        class_name = model.names[class_id]

        # Format the result as 'nameOfObject,boundary_Type(rectangle),coordinates'
        result = f"{class_name},rectangle,{x1},{y1},{x2},{y2}\n"

        # Write the result to the text file
        f.write(result)

print("Detection results saved to 'detection_results.txt'")
