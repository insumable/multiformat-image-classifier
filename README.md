# MultiFormat Image Classifier

A Computer Vision project that uses a Convolutional Neural Network (CNN) built with PyTorch to classify images into multiple visual content categories. The model is trained on a curated dataset containing paintings, photographs, sketches, and digital artwork.

Input          : 3 x 128 x 128

Conv32         : 32 x 128 x 128
MaxPool        : 32 x 64 x 64

Conv64         : 64 x 64 x 64
MaxPool        : 64 x 32 x 32

Conv128        : 128 x 32 x 32
MaxPool        : 128 x 16 x 16

Flatten        : 32768

Dense128       : 128
Output         : 5

## Features

- Image preprocessing and augmentation
- CNN architecture implemented in PyTorch
- Multi-class image classification
- Model training and validation
- Performance evaluation
- Prediction on custom images


- Python
- PyTorch
- TorchVision
- NumPy
- Pandas
- Matplotlib
- OpenCV

## Dataset Categories

- Paintings
- Photographs
- Sketches
- Digital Artwork

## Project Structure

├── data/
├── notebooks/
├── models/
├── training/
├── inference/
├── results/
└── README.md

## Results

The CNN automatically learns hierarchical visual features and performs multi-class classification on diverse image categories.
