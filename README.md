# walk-of-life-recognizer

Deep learning model to identify runner photos for Walk of Life 2026 in Naples.

## Overview

This project provides a Google Colab notebook that fine-tunes a **pretrained
ResNet-18** (PyTorch / ImageNet weights) to classify photos as *runner* or
*non-runner* from the Walk of Life 2026 event.

## Getting Started

### 1. Prepare the dataset on Google Drive

Upload your images to Google Drive in the following folder structure:

```
My Drive/
  walk_of_life_dataset/
    train/
      runner/        ← training images of runners
      non_runner/    ← training images of non-runners
    val/
      runner/        ← validation images of runners
      non_runner/    ← validation images of non-runners
```

### 2. Open the notebook in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moriarty2002/walk-of-life-recognizer/blob/main/walk_of_life_recognizer.ipynb)

Or upload `walk_of_life_recognizer.ipynb` manually to
[Google Colab](https://colab.research.google.com/).

### 3. Select a GPU runtime

Go to **Runtime → Change runtime type** and choose **T4 GPU** (or any
available GPU) for faster training.

### 4. Run all cells

The notebook will:

1. Mount your Google Drive.
2. Load and augment the dataset.
3. Fine-tune a pretrained ResNet-18 model.
4. Evaluate performance on the validation set.
5. Save the trained model back to Google Drive.

## Project Structure

```
walk-of-life-recognizer/
├── walk_of_life_recognizer.ipynb   # Main Colab notebook
└── README.md
```

## Requirements

All dependencies are pre-installed on Google Colab:

- Python 3
- PyTorch & torchvision
- matplotlib, numpy, scikit-learn, Pillow
