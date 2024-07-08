# Task-3-NullClass

# Animal Classification Project

This project aims to classify images of animals into herbivores and carnivores using machine learning techniques. The model is trained on a dataset of animal images, leveraging convolutional neural networks (CNNs) for image classification.


## Introduction

The Animal Classification Project utilizes deep learning to classify images of animals based on their dietary habits. This repository contains the code for training the model, evaluating its performance, and a graphical user interface (GUI) for interactive image classification.

## Features

- **Image Classification**: Classify animal images into herbivores and carnivores.
- **GUI**: Provides a user-friendly interface for uploading images and getting classification results.
- **Model Evaluation**: Evaluate model performance on test datasets.
- **Data Augmentation**: Utilize data augmentation techniques to enhance model robustness.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/animal-classification.git
   cd animal-classification
   ```

2. **Install dependencies**:

   Ensure you have Python 3.x and pip installed. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare the dataset**:

   - Download the Animals-10 dataset from [Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10).
   - Extract the dataset and place it in a directory (`data/` by default).

## Usage

1. **Training the Model**:

   ```bash
   python train_model.py
   ```

2. **Launching the GUI**:

   ```bash
   python gui.py
   ```

   The GUI will open, allowing you to upload images and get real-time classification results.

## Dataset

The model is trained on the Animals-10 dataset, consisting of images categorized into 10 classes of animals. The dataset includes both herbivores and carnivores, providing a diverse set of images for training and evaluation.

- Dataset: [Animals-10 on Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

## Model Architecture

The model architecture used for this project is based on convolutional neural networks (CNNs), designed to extract features from animal images and classify them into two classes: herbivores and carnivores.

- Architecture: Convolutional Neural Network (CNN)
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam Optimizer
- Metrics: Accuracy

## Results

The trained model achieves an accuracy of 87% on the validation dataset, demonstrating its capability to classify animal images based on dietary habits effectively.

