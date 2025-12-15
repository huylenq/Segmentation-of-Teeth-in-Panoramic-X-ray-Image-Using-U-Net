# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:03:25 2020

@author: serdarhelli

Batch Connected Component Analysis for dental X-ray segmentation.

Usage:
    python OPENCV_cca.py --images <images_dir> --predictions <predictions_dir> --output <output_dir>

    Or import and call run_batch_cca() directly.
"""

import argparse
import cv2
import numpy as np
from imutils import perspective
import os
from pathlib import Path
from scipy.spatial import distance as dist


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# Morphological kernels
KERNEL_5x5 = np.ones((5, 5), dtype=np.float32)
KERNEL_SHARPENING = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])


def run_batch_cca(images_dir, predictions_dir, output_dir, erode_iterations=2, open_iterations=3):
    """
    Run batch CCA analysis on predicted segmentation masks.

    Args:
        images_dir: Directory containing original images
        predictions_dir: Directory containing predicted mask images
        output_dir: Directory to save CCA-processed results
        erode_iterations: Number of erosion iterations (default: 2)
        open_iterations: Number of morphological opening iterations (default: 3)
    """
    images_dir = Path(images_dir)
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted(os.listdir(predictions_dir), key=len)

    for filename in dirs:
        # Load predicted mask and original image
        # Names must match between predictions and original images
        image = cv2.imread(str(predictions_dir / filename))
        image2 = cv2.imread(str(images_dir / filename))

        if image is None or image2 is None:
            print(f"Skipping {filename}: could not load image pair")
            continue

        # Morphological processing
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, KERNEL_5x5, iterations=open_iterations)
        image = cv2.filter2D(image, -1, KERNEL_SHARPENING)
        image = cv2.erode(image, KERNEL_5x5, iterations=erode_iterations)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        labels = cv2.connectedComponents(thresh, connectivity=8)[1]
        unique_labels = np.unique(labels)
        count2 = 0

        for label in unique_labels:
            if label == 0:
                continue

            # Create a mask for this component
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # Find contours and determine contour area
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0]
            c_area = cv2.contourArea(cnts)

            # Threshold for tooth count
            if c_area > 2000:
                count2 += 1

            rect = cv2.minAreaRect(cnts)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            color1 = list(np.random.choice(range(150), size=3))
            color = [int(color1[0]), int(color1[1]), int(color1[2])]
            cv2.drawContours(image2, [box.astype("int")], 0, color, 2)
            (tl, tr, br, bl) = box

            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Draw the midpoints on the image
            cv2.circle(image2, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(image2, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(image2, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(image2, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv2.line(image2, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), color, 2)
            cv2.line(image2, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), color, 2)

            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            pixelsPerMetric = 1
            dimA = dA * pixelsPerMetric
            dimB = dB * pixelsPerMetric
            cv2.putText(image2, "{:.1f}px".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(image2, "{:.1f}px".format(dimB), (int(trbrX + 10), int(trbrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(image2, "{:.0f}".format(label), (int(tltrX - 35), int(tltrY - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        print(f"Image: {filename}, Tooth count: {count2}")
        cv2.imwrite(str(output_dir / filename), image2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch CCA analysis for dental X-ray segmentation")
    parser.add_argument("--images", required=True, help="Directory containing original images")
    parser.add_argument("--predictions", required=True, help="Directory containing predicted masks")
    parser.add_argument("--output", required=True, help="Directory to save CCA results")
    parser.add_argument("--erode", type=int, default=2, help="Erosion iterations (default: 2)")
    parser.add_argument("--open", type=int, default=3, help="Opening iterations (default: 3)")

    args = parser.parse_args()
    run_batch_cca(args.images, args.predictions, args.output, args.erode, args.open)
