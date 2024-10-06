import time
import cv2
import torch
import numpy as np
import argparse
import os

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 Vehicle Detection with Disability Symbol")
    parser.add_argument('--video', type=str, help='Path to the video file', required=True)
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to save the output video')
    parser.add_argument('--show-original', action='store_true', help='Flag to show the original bounding boxes as well')
    args = parser.parse_args()
    return args

# Call the function to get the arguments
args = parse_args()

# Now call the detect_and_display function with the passed arguments
video_path = args.video
save_path = args.output
show_original = args.show_original

# Device selection: Check if GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and apply half precision if using GPU
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../yolov5/ft_models/both_5n.pt').to(device)
if device.type == 'cuda':
    model.half()  # Apply half precision only if using GPU

print(f'Model loaded and running on {device}')

# Resize frame to be divisible by 32
def resize_to_multiple_of_32(frame):
    height, width = frame.shape[:2]
    new_height = int(np.ceil(height / 32) * 32)
    new_width = int(np.ceil(width / 32) * 32)
    return cv2.resize(frame, (new_width, new_height))

# Post-process the output tensor to extract bounding boxes and confidence scores
def process_results(results):
    boxes = []
    confidences = []
    classes = []
    
    # Assuming results is a [batch_size, num_predictions, 6] tensor where:
    # - 6 corresponds to [x1, y1, x2, y2, confidence, class]
    results = results.squeeze(0)  # Remove the batch dimension
    for result in results:
        x1, y1, x2, y2, conf, cls = result[:6].tolist()
        boxes.append((x1, y1, x2, y2))
        confidences.append(conf)
        classes.append(int(cls))
    
    return boxes, confidences, classes

# Check if the symbol is inside the car
def is_symbol_in_car(car_bbox, symbol_bbox):
    car_x1, car_y1, car_x2, car_y2 = car_bbox[:4]
    sym_x1, sym_y1, sym_x2, sym_y2 = symbol_bbox[:4]
    
    return car_x1 < sym_x1 < car_x2 and car_y1 < sym_y1 < car_y2

# Main function to handle video processing
def detect_and_display(video_path, save_path, show_original):
    # Open the video source
    cap = cv2.VideoCapture(video_path)
    fps = 30  # Desired FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_idx = 0
    headless = False  # Force display mode
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break  # Break when no more frames

        # Resize the frame to be divisible by 32
        resized_frame = resize_to_multiple_of_32(frame)

        # Convert frame to tensor, add a batch dimension, and move to the correct device
        frame_tensor = torch.from_numpy(resized_frame).to(device).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension and permute channels
        frame_tensor = frame_tensor.float() / 255.0  # Normalize the frame
        
        if device.type == 'cuda':
            frame_tensor = frame_tensor.half()  # Convert to half precision if using GPU

        # Perform detection on the frame
        with torch.no_grad():
            results = model(frame_tensor)  # Get raw model outputs
        
        # Post-process results
        boxes, confidences, classes = process_results(results)

        cars = []
        symbols = []

        for i, cls in enumerate(classes):
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            if cls == 0:  # Assuming class 0 is 'car'
                cars.append((x1, y1, x2, y2, conf))
            elif cls == 1:  # Assuming class 1 is 'symbol_of_access'
                symbols.append((x1, y1, x2, y2, conf))

        # Draw the original bounding boxes if requested
        if show_original:
            for car_bbox in cars:
                car_x1, car_y1, car_x2, car_y2, car_conf = car_bbox
                cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (255, 0, 0), 2)  # Blue for car
                cv2.putText(frame, f"Car: {car_conf:.2f}", (int(car_x1), int(car_y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for symbol_bbox in symbols:
                sym_x1, sym_y1, sym_x2, sym_y2, sym_conf = symbol_bbox
                cv2.rectangle(frame, (int(sym_x1), int(sym_y1)), (int(sym_x2), int(sym_y2)), (0, 0, 255), 2)  # Red for symbol
                cv2.putText(frame, f"Symbol: {sym_conf:.2f}", (int(sym_x1), int(sym_y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Check if any symbol of access lies within a car's boundary and draw the new bounding box
        for car_bbox in cars:
            car_x1, car_y1, car_x2, car_y2, car_conf = car_bbox
            for symbol_bbox in symbols:
                sym_x1, sym_y1, sym_x2, sym_y2, sym_conf = symbol_bbox
                if is_symbol_in_car(car_bbox, symbol_bbox):
                    combined_conf = car_conf * sym_conf  # Calculate combined confidence
                    # Draw the new bounding box around the car
                    cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 2)  # Green for combined box
                    cv2.putText(frame, f"Combined Conf: {combined_conf:.2f}", (int(car_x1), int(car_y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display processing time on the frame
        processing_time_ms = (time.time() - start_time) * 1000  # in milliseconds
        cv2.putText(frame, f"Time: {int(processing_time_ms)} ms", (frame_width - 200, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Write the frame to the output video file
        out.write(frame)

        if not headless:
            # Show the frame in real-time with OpenCV
            cv2.imshow('Frame', frame)
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
print('Detect and display')
detect_and_display(video_path, save_path, show_original)