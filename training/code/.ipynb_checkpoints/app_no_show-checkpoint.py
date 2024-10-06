import time
import cv2
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


# Add this before the detect_and_display function
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Vehicle Detection with Disability Symbol")
    parser.add_argument('--video', type=str, help='Path to the video file')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to save the output video')
    args = parser.parse_args()
    return args

# Call the function to get the arguments
args = parse_args()

# Now call the detect_and_display function with the passed arguments
video_path = args.video
save_path = args.output

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/work/users/jerryma/yolov5/ft_models/both_5n.pt')
print('model loaded: ', type(model))


def detect_and_display(video_path, save_path):
    # Open the video source
    cap = cv2.VideoCapture(video_path)
    fps = 30  # Desired FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    frame_idx = 0
    headless = not os.getenv('DISPLAY')  # Check if display is available (headless environment)
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break  # Break when no more frames

        # Perform detection on the frame
        results = model(frame)

        # Get detected objects (bounding boxes and class IDs)
        objects = results.xyxy[0]  # xyxy format (x1, y1, x2, y2, confidence, class)

        # Separate cars and symbols
        cars = []
        symbols = []

        for obj in objects:
            x1, y1, x2, y2, conf, cls = obj
            if int(cls) == 0:  # Assuming class 0 is 'car'
                cars.append((x1, y1, x2, y2, conf))
            elif int(cls) == 1:  # Assuming class 1 is 'symbol_of_access'
                symbols.append((x1, y1, x2, y2, conf))

        # Check if any symbol of access lies within a car's boundary
        for car_bbox in cars:
            car_x1, car_y1, car_x2, car_y2, car_conf = car_bbox
            for symbol_bbox in symbols:
                sym_x1, sym_y1, sym_x2, sym_y2, sym_conf = symbol_bbox

                # Check if symbol's bounding box lies within the car's bounding box
                if car_x1 < sym_x1 and car_y1 < sym_y1 and car_x2 > sym_x2 and car_y2 > sym_y2:
                    # Draw the car bounding box
                    cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 2)

        # Display processing time on the frame
        processing_time_ms = (time.time() - start_time) * 1000  # in milliseconds
        cv2.putText(frame, f"Time: {int(processing_time_ms)} ms", (frame_width - 200, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Write the frame to the output video file
        out.write(frame)

        if not headless:
            # Show the frame in real-time only if display is available
            # Replace cv2.imshow with this code to display the frame using matplotlib
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()
            key = cv2.waitKey(max(1, int(1000 / fps - processing_time_ms)))  # Adjust waitKey based on processing time
            if key == ord('q'):
                break

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    if not headless:
        cv2.destroyAllWindows()
# Example usage
print('detect and display')
detect_and_display(video_path, save_path)