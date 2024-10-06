from ultralytics import YOLO
import cv2
import math 
import torch

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path= '../yolov5/ft_models/both_5n.pt').to(device)

# Object classes
classNames = ["cars", "symbol_of_access"]

while True:
    success, img = cap.read()
    results = model(img)  # Removed stream=True

    # Coordinates
    boxes = results.xyxy[0]  # Access the first batch of results

    for box in boxes:
        # Bounding box
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Confidence
        confidence = math.ceil((box[4].item() * 100)) / 100
        print("Confidence --->", confidence)

        # Class name
        cls = int(box[5].item())
        print("Class name -->", classNames[cls])

        # Set color based on class
        if classNames[cls] == "cars":
            color = (128, 0, 128)  # Purple
        else:
            color = (173, 216, 230)  # Light blue

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Display class name and confidence
        label = f"{classNames[cls]} {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()