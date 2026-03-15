# Camouflage Object Detection System

## What this project does
Detects camouflaged soldiers hidden in forest environments using a 
USB webcam in real time. Draws bounding boxes around detected soldiers 
and estimates distance. Built to run on Raspberry Pi 5 with zero budget.

## Why YOLOv8 Nano
YOLOv8 nano was chosen specifically for Raspberry Pi 5's limited 
computing power. Larger models are more accurate but too slow for 
real-time detection on low-power hardware — nano gives the best 
speed/accuracy tradeoff on edge devices.

## Technical approach
- Transfer learning from COCO pretrained weights
- Fine-tuned on camouflage-specific dataset for 100 epochs
- Early stopping at patience=20 to prevent wasted training
- Data augmentation for varied soldier positions and lighting
- Confidence threshold set to 0.25 to minimize missed detections

## Results
- Real-time detection on Raspberry Pi 5
- Streamlit dashboard for live visualization
- Bounding box detection with distance estimation

## Tools used
Python, YOLOv8 (Ultralytics), PyTorch, Streamlit, Raspberry Pi 5