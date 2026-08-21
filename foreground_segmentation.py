"""
Foreground Segmentation Toolkit
--------------------------------
A thin, reusable wrapper around OpenCV's background subtraction algorithms
(MOG2, KNN) with shadow removal, morphological cleanup, and a simple CLI
for running on video files, image sequences, or a live webcam.

References:
    Z. Zivkovic, "Improved adaptive Gaussian mixture model for background
    subtraction," ICPR 2004.
    Z. Zivkovic and F. van der Heijden, "Efficient adaptive density estimation
    per image pixel for the task of background subtraction," Pattern
    Recognition Letters, 2006.
"""

import argparse
import sys

import cv2
import numpy as np


class ForegroundSegmenter:
    """Wraps an OpenCV background subtractor with sensible defaults and
    post-processing (shadow removal + morphological noise cleanup).
    """

    def __init__(self, method="mog2", detect_shadows=True, history=500,
                 var_threshold=16, morph_kernel_size=5):
        method = method.lower()
        if method == "mog2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )
        elif method == "knn":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=400.0,
                detectShadows=detect_shadows,
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'mog2' or 'knn'.")

        self.method = method
        self.detect_shadows = detect_shadows
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    def apply(self, frame):
        """Return a clean binary foreground mask (0/255) for a single frame."""
        raw_mask = self.subtractor.apply(frame)

        # Shadows are labeled 127 by OpenCV when detect_shadows=True; drop them.
        if self.detect_shadows:
            _, mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            mask = raw_mask

        # Morphological cleanup: remove speckle noise, close small gaps.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def extract_foreground(self, frame):
        """Return the frame with background pixels zeroed out."""
        mask = self.apply(frame)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def overlay(self, frame, color=(0, 255, 0), alpha=0.4):
        """Return the frame with foreground regions highlighted in `color`."""
        mask = self.apply(frame)
        colored = np.zeros_like(frame)
        colored[:] = color
        colored_mask = cv2.bitwise_and(colored, colored, mask=mask)
        return cv2.addWeighted(frame, 1.0, colored_mask, alpha, 0)


def _parse_args():
    parser = argparse.ArgumentParser(description="Foreground Segmentation Toolkit")
    parser.add_argument("--input", required=True,
                         help="Path to video file, or '0' for default webcam")
    parser.add_argument("--output", default=None,
                         help="Path to save output video (optional)")
    parser.add_argument("--method", default="mog2", choices=["mog2", "knn"],
                         help="Background subtraction method")
    parser.add_argument("--mode", default="foreground",
                         choices=["mask", "foreground", "overlay"],
                         help="What to output: binary mask, foreground-only, or overlay")
    parser.add_argument("--no-shadows", action="store_true",
                         help="Disable shadow detection")
    parser.add_argument("--live", action="store_true",
                         help="Display output in a live window")
    return parser.parse_args()


def main():
    args = _parse_args()

    source = 0 if args.input == "0" else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open input '{args.input}'", file=sys.stderr)
        sys.exit(1)

    segmenter = ForegroundSegmenter(
        method=args.method, detect_shadows=not args.no_shadows
    )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.mode == "mask":
            result = segmenter.apply(frame)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif args.mode == "overlay":
            result = segmenter.overlay(frame)
        else:
            result = segmenter.extract_foreground(frame)

        if writer:
            writer.write(result)

        if args.live:
            cv2.imshow("Foreground Segmentation", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
