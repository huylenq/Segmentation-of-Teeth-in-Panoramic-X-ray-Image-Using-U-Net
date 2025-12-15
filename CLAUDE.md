# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CRITICAL: all dependencies are isolated in .venv, install using pip.

## Project Overview

This is a reference implementation for semantic segmentation of teeth in panoramic X-ray images using U-Net. Based on the paper "Tooth Instance Segmentation on Panoramic Dental Radiographs Using U-Nets and Morphological Processing" by Helli & Hamamci.

**Dataset:** Dental panoramic X-rays from Mendeley (116 images, 512x512 resolution after preprocessing).

## Quick Start

Run `Main.ipynb` in Google Colab for the complete training pipeline. The notebook handles:
1. Dataset download from Mendeley
2. Data preprocessing and augmentation (Albumentations)
3. U-Net training (TensorFlow/Keras)
4. Inference and post-processing (Connected Component Analysis)

## Architecture

```
├── model.py              # U-Net architecture (TensorFlow/Keras)
├── download_dataset.py   # Dataset fetcher from Mendeley
├── images_prepare.py     # Image loading and resizing
├── masks_prepare.py      # Mask preparation (original or pre-split)
├── CCA_Analysis.py       # Post-processing: tooth counting and measurement
├── Binary_Image_Analysis/
│   ├── OPENCV_cca.py     # Batch CCA processing
│   └── optional_watershed.py
├── Scores_and_Test/
│   └── Utest.py          # Mann-Whitney U statistical tests
├── Custom_Masks/         # Pre-split 512x512 masks (zipped)
└── Original_Masks/       # Original resolution masks (zipped)
```

## Key Components

### U-Net Model (`model.py`)
- Input: 512x512x1 grayscale
- Encoder: 5 levels (32→64→128→256→512 filters)
- Features: BatchNorm after each conv block, Dropout (0.1→0.5), He initialization
- Output: Sigmoid activation for binary segmentation

### Data Pipeline
- `pre_images()`: Load and resize X-ray images, handle RGB→grayscale conversion
- `pre_masks()` / `pre_splitted_masks()`: Load segmentation masks
- Augmentation: Albumentations (RandomCrop, brightness/contrast, rotation, scale, blur, noise)

### Post-Processing (`CCA_Analysis.py`)
Connected Component Analysis pipeline:
1. Morphological opening and sharpening
2. Erosion to separate touching teeth
3. `cv2.connectedComponents()` for tooth isolation
4. Bounding box fitting with `cv2.minAreaRect()`
5. Tooth counting (area threshold: 2000 pixels)
6. Dimension measurement in pixels

## Dependencies

TensorFlow 2.4+, OpenCV, scikit-image, Albumentations, imutils, natsort.

Note: OpenCV version conflicts with Albumentations may require: `pip install opencv-python-headless==4.5.2.52`

## Training Parameters

Default configuration in `Main.ipynb`:
- Train/test split: 105/11 images
- Augmentation: 4x data expansion
- Loss: binary_crossentropy
- Optimizer: Adam
- Batch size: 8
- Epochs: 200
- Threshold for binarization: 0.25

## Statistical Validation

`Scores_and_Test/Utest.py` performs Mann-Whitney U tests comparing:
- Tooth count error: non-processed vs post-processed predictions
- Dice scores across 10-fold cross-validation
