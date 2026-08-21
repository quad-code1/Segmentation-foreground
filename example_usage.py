"""
Minimal example: run foreground segmentation on a webcam feed and
display the result live.

Usage:
    python examples/example_usage.py
"""

import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.foreground_segmentation import ForegroundSegmenter


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Try a video file path instead.")
        return

    segmenter = ForegroundSegmenter(method="mog2", detect_shadows=True)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        foreground = segmenter.extract_foreground(frame)
        cv2.imshow("Original", frame)
        cv2.imshow("Foreground", foreground)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
