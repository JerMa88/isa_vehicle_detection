README
Overview
This repository contains a project for vehicle detection using YOLOv5. The project includes training a custom YOLOv5 model by fine-tuning it on a specific dataset and running the detection algorithm on video files.

Training by Fine-Tuning YOLOv5
Prerequisites
Python 3.6+
PyTorch
OpenCV
YOLOv5 repository
Steps
1. Clone the YOLOv5 Repository:
(refer to setup.md)

2. install neccessary requirements
(refer to setup.md)

3. Prepare the Dataset:

Organized dataset in the following structure for yolov5:
```dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
```
Each image should have a corresponding label file in YOLO format.

4. Train the Model:
Create a custom YAML configuration file for your dataset (e.g., custom_dataset.yaml):
```
train: ../dataset/images/train
val: ../dataset/images/val
nc: 2  # number of classes
names: ['cars', 'symbol_of_access']
```
Run the training command:

```python train.py --img 640 --batch 16 --epochs 50 --data custom_dataset.yaml --weights yolov5s.pt```

Running the Detection Algorithm
Prerequisites
Python 3.6+
PyTorch
OpenCV

Steps

1. Run the Detection Script:

The detection script (app_video.py) processes a video file and outputs the processed video with detected bounding boxes and confidence scores.
Example command to run the script:

```python app.py --video ../example_data/video/ISA\ Parking\ Placards.mp4 --output ./original_output_video.mp4 --show-original```
The script reads a video file frame by frame, processes each frame using the YOLOv5 model, and draws bounding boxes around detected objects. The processing time for each frame is displayed in the bottom-right corner of the video.

The script processes each frame, detects objects, and draws bounding boxes with different colors based on the class of the detected object.

Conclusion
This project demonstrates how to fine-tune a YOLOv5 model for custom object detection and run the detection algorithm on video files. The script processes each frame in real-time, displays the processing time, and saves the processed video with detected bounding boxes.

 





