import argparse
import cv2
import math
import time
import torch
from ultralytics import YOLO

def main(video_path, output_path, show_original):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/ft_models/both_5n.pt').to(device)

    # Object classes
    classNames = ["cars", "symbol_of_access"]

    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        results = model(frame)

        # Coordinates
        boxes = results.xyxy[0]  # Access the first batch of results

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Confidence
            confidence = math.ceil((box[4].item() * 100)) / 100

            # Class name
            cls = int(box[5].item())

            # Set color based on class
            if classNames[cls] == "cars":
                color = (128, 0, 128)  # Purple
            else:
                color = (173, 216, 230)  # Light blue

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Display class name and confidence
            label = f"{classNames[cls]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Set color based on processing time
        if processing_time > 33.3:
            time_color = (0, 0, 255)  # Red
        elif processing_time > 30.0:
            time_color = (0, 255, 255)  # Yellow
        else:
            time_color = (0, 255, 0)  # Green

        cv2.putText(frame, f"Processing Time: {processing_time:.2f} ms", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, time_color, 2)

        if show_original:
            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video file with YOLO model.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output video file.')
    parser.add_argument('--show-original', action='store_true', help='Show the original video with detections in real-time.')

    args = parser.parse_args()
    main(args.video, args.output, args.show_original)