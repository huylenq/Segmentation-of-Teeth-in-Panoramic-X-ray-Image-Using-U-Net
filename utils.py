from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


def load_and_preprocess(image_path: Path, target_size: tuple = (512, 512)) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess an X-ray image for inference.

    Returns:
        preprocessed: (1, 512, 512, 1) float32 array normalized to [0, 1]
        original: Original image as numpy array
    """
    img = Image.open(image_path)
    original = np.array(img)

    # Resize to model input size
    img_resized = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img_resized)

    # Convert to grayscale if needed
    if len(img_array.shape) > 2:
        img_array = img_array[:, :, 0]

    # Normalize and reshape
    preprocessed = img_array.astype(np.float32) / 255.0
    preprocessed = preprocessed.reshape(1, target_size[0], target_size[1], 1)

    return preprocessed, original


def visualize_prediction(original, raw_prediction, binary_mask):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original X-ray', fontsize=14)
    axes[0].axis('off')

    # Raw prediction (probability map)
    im = axes[1].imshow(raw_prediction, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Prediction (Probability)', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Binary Mask (threshold=0.25)', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_overlay(image_path, original, raw_prediction):
    prediction_resized = cv2.resize(raw_prediction, (original.shape[1], original.shape[0]),
                                    interpolation=cv2.INTER_LANCZOS4)

    # Create binary mask at original resolution
    mask_uint8 = np.uint8(prediction_resized * 255)
    _, mask_binary = cv2.threshold(mask_uint8, thresh=255 // 2, maxval=255, type=cv2.THRESH_BINARY)

    # Find and draw contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create overlay image
    if len(original.shape) == 2:
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original.copy()

    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Display
    plt.figure(figsize=(14, 10))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f'Segmentation Overlay - {image_path.name}', fontsize=16)
    plt.axis('off')
    plt.show()