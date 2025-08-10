# Cataract Detection Using ResNet18

This project implements a cataract eye disease classifier using a fine-tuned ResNet18 model in PyTorch. It classifies images into two categories: **cataract** and **normal**.

## Dataset

The dataset used for training and testing is from Kaggle:

[Cataract Dataset on Kaggle](https://www.kaggle.com/datasets/jr2ngb/cataractdataset)

## Features

- Data augmentation and normalization for robust training
- Handles class imbalance with weighted loss
- Early stopping to prevent overfitting
- Training, validation, and testing pipeline
- Visualization of misclassified images

## Requirements

- Python 3.6+
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn

Install dependencies with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
