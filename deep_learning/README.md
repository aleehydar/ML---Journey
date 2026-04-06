# CIFAR-10 Image Classifier — PyTorch CNN

## What this project does
A Convolutional Neural Network trained to classify images into 10 categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
The model applies convolutional filters to detect edges and patterns, uses
MaxPooling to reduce spatial dimensions, then passes features through fully
connected layers to output confidence scores for each class.

## Architecture
- Conv1: 3 → 32 filters (3×3, padding=1) + ReLU + MaxPool
- Conv2: 32 → 64 filters (3×3, padding=1) + ReLU + MaxPool  
- Flatten: 64 × 8 × 8 = 4096 features
- FC1: 4096 → 512 neurons + ReLU
- FC2: 512 → 10 class outputs

## Results
| Metric | Value |
|--------|-------|
| Dataset | CIFAR-10 (50,000 train / 10,000 test) |
| Epochs | 5 |
| Test Accuracy | 72% |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |

## Tools used
Python, PyTorch, torchvision