# CNN-Based Image Classification with PyTorch

A convolutional neural network that classifies images into five visual categories — **Painting, Photo, Schematics, Sketch, and Text** — built, trained, and evaluated end-to-end in PyTorch.

**Result: 95.05% test accuracy** on 2,932 held-out images.

## Link

link : https://www.kaggle.com/code/sohampatra2/image-classification

## Overview

This notebook covers the full pipeline for a 5-class image classifier:

1. Load a raw image dataset and index it into a single DataFrame
2. Drop low-quality images (blurry / low-resolution / near-duplicate) using metadata from a companion data-QA pipeline
3. Preprocess images for PyTorch (resize, normalize, tensorize)
4. Train a CNN from scratch
5. Evaluate with accuracy, a confusion matrix, and a per-class classification report

## Dataset

| | |
|---|---|
| Source | Kaggle dataset `ranimbenamara/dataset-livrable-1` |
| Raw images | 16,705 |
| After cleaning | 14,657 |
| Train / test split | 11,725 / 2,932 (80 / 20, stratified by label) |

Raw class distribution:

| Class | Images |
|---|---|
| Photo | 5,338 |
| Schematics | 4,902 |
| Text | 4,874 |
| Sketch | 1,379 |
| Painting | 212 |

> **Class imbalance**: Painting accounts for just over 1% of the dataset. This shows up directly in the results below.

## Preprocessing

Implemented in the `inputdata` PyTorch `Dataset` class:

- BGR → RGB conversion (OpenCV)
- Resize to 128 × 128
- Scale to [0, 1], then normalize with ImageNet mean/std
- Convert to a `(C, H, W)` tensor

## Model Architecture

```
Input (3 × 128 × 128)
  → Conv2d(3→32, 3×3) + ReLU + MaxPool(2×2)    → 32  × 64 × 64
  → Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)   → 64  × 32 × 32
  → Conv2d(64→128, 3×3) + ReLU + MaxPool(2×2)  → 128 × 16 × 16
  → Flatten → Linear(32768→128) + ReLU + Dropout(0.5)
  → Linear(128→5)
```

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 20 |
| Device | CUDA if available, else CPU |

Training loss fell steadily from **0.461** (epoch 1) to **0.041** (epoch 20).

## Results

**Test accuracy: 95.05%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Painting | 0.40 | 0.15 | 0.21 | 41 |
| Photo | 0.94 | 0.95 | 0.94 | 1,066 |
| Schematics | 0.92 | 0.94 | 0.93 | 753 |
| Sketch | 0.97 | 1.00 | 0.98 | 263 |
| Text | 0.99 | 1.00 | 0.99 | 809 |
| **Accuracy** | | | **0.95** | 2,932 |
| Macro avg | 0.84 | 0.80 | 0.81 | 2,932 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 2,932 |


## Key Findings

- The model is strong overall, especially on **Sketch** and **Text** (F1 ≥ 0.98).
- **Painting is the clear weak point** (recall 0.15) — of 41 painting test images, only 6 were classified correctly; 31 were confused with **Photo**.
- This tracks directly with the dataset imbalance (212 raw Painting images vs. thousands per other class), and likely reflects genuine visual overlap between paintings and photos/sketches depending on style.

## Possible Improvements

- Class-balancing via oversampling, a weighted loss, or targeted augmentation for Painting
- General data augmentation (flips, crops, color jitter) for better generalization
- A deeper backbone or transfer learning (e.g. ResNet) for richer features
- A validation split with LR scheduling / early stopping

## Project Structure

```
.
├── image-classification.ipynb   # full pipeline: load → clean → train → evaluate
├── assets/
│   └── confusion_matrix.png
└── README.md
```

## Getting Started

1. Clone the repo and open `image-classification.ipynb`. It was originally run as a Kaggle notebook, so data paths are under `/kaggle/input/...`.
2. If running elsewhere, update `Base_Path` and `Meta_Path` to point to your local copies of the image dataset and the QA metadata CSVs (`blurry_images.csv`, `low_resolution_images.csv`, `near_duplicate_images.csv`, `clip_corrected_labels.csv`).
3. Install the dependencies below and run all cells in order.

## Requirements

```bash
pip install torch numpy pandas opencv-python matplotlib seaborn scikit-learn kagglehub
```

## Acknowledgments

- Image dataset: Kaggle — `ranimbenamara/dataset-livrable-1`
- Data quality & annotation metadata: Kaggle — `sohampatra2/data-preparation-and-annotation`
