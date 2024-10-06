# ISA Vehicle Detection

**ISA Vehicle Detection** is a computer vision project that leverages machine learning to detect vehicles displaying the International Symbol of Access (ISA), enabling efficient guidance to accessible parking spaces.

## Table of Contents

- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Ran Into](#challenges-we-ran-into)
- [Accomplishments That We're Proud Of](#accomplishments-that-were-proud-of)
- [What We Learned](#what-we-learned)
- [What's Next for ISA Vehicle Detection](#whats-next-for-isa-vehicle-detection)

## Inspiration

Accessibility in parking, especially at crowded concert and game venues, is a significant concern for individuals with disabilities. The ParkHub challenge to identify vehicles displaying the International Symbol of Access (ISA) intrigued our team. We wanted to explore more about computer vision and machine learning while creating a tool that can help streamline the parking process even further, saving time and money for both venues and the people that attend them.

## What It Does

Our project leverages computer vision and machine learning to detect vehicles that display the ISA. The model processes images to identify cars and the ISA emblem. By analyzing the overlap of bounding boxes around detected cars and symbols, the model can accurately determine which vehicles require accessible parking. This allows ParkHub to direct these vehicles to their designated parking areas efficiently.

## How We Built It

We first began by sourcing datasets that contained images of cars and the International Symbol of Access (ISA). Since we could not find any datasets with both labels identified, we merged multiple datasets and used **Roboflow** to manually label over 2,000 images to ensure accuracy. This was very time-consuming but was a crucial step in our process.

When we were training and testing our initial model, we noticed it performed well on stock images but struggled with real-world video footage. To address this, we expanded our dataset by collecting real-world training and testing images. This hands-on approach improved our dataset and significantly enhanced the model's performance in practical scenarios.

For the model development, we utilized the **YOLOv5** object detection algorithms. We specifically trained the **nano version** for its balance of speed and accuracy, making it suitable for real-time applications and especially videos. Additionally, we trained the **small** and **medium** versions of the YOLOv5 model to compare their performance and accuracy levels. By fine-tuning these models, we improved their ability to accurately identify overlapping bounding boxes of cars and the ISA, enhancing the detection of accessible vehicles.

## Challenges We Ran Into

One of the main challenges was the lack of existing datasets that included both cars and the ISA labels. This made us go through the manual process of labeling the images, which was a time-consuming task. Additionally, since most of our dataset consisted of stock images, it did not provide the variability needed for real-world application. Therefore, it was essential to spend time collecting our own data to improve model robustness. Ensuring the model could accurately interpret overlapping bounding boxes to identify ISA vehicles added another layer of complexity to the project.

## Accomplishments That We're Proud Of

We're proud of our resilience and perseverance in solving the data scarcity issue by creating a comprehensive, well-labeled dataset. Collecting real-world images took a lot of time but was well worth it since it significantly enhanced the model's accuracy. Successfully integrating the detection of both vehicles and the ISA symbol to accurately identify accessible vehicles in 24 hours is something we are proud that we accomplished.

## What We Learned

This project highlighted the critical role of high-quality data in machine learning. We learned the importance of data diversity and real-world applicability in training effective models. The experience taught us valuable lessons in dataset creation, annotation, and the different elements of object detection algorithms. It also highlighted the challenges and rewards of developing solutions with real-world impact.

## What's Next for ISA Vehicle Detection

Moving forward, we aim to further refine our model by expanding the dataset with more diverse and real-world images. Additionally, exploring the use of advanced techniques such as deep learning-based segmentation like **SAM** could enhance the precision of ISA detection.

