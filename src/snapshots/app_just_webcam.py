from ultralytics import YOLO
import cv2
import math 
import torch
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = torch.hub.load('ultralytics/yolov5', 'custom', path= '../yolov5/ft_models/both_5n.pt').to(device)

# object classes
classNames = ["cars", "symbol_of_access"]

while True:
    success, img = cap.read()
    results = model(img)  # Removed stream=True

    # coordinates
    boxes = results.xyxy[0]  # Access the first batch of results

    for box in boxes:
        # bounding box
        x1, y1, x2, y2 = box[:4].int()  # convert to int values

        # put box in cam
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # confidence
        confidence = math.ceil((box[4].item() * 100)) / 100
        print("Confidence --->", confidence)

        # class name
        cls = int(box[5].item())
        print("Class name -->", classNames[cls])

        # object details
        org = (x1, y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()