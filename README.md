# Foreground Segmentation Toolkit

A lightweight, easy-to-use toolkit for foreground/background segmentation in video and image sequences, built on top of OpenCV's background subtraction algorithms (MOG2, KNN) with a clean Python API and CLI.

## What this is

This project is a novel research contribution utility layer around well-established, peer-reviewed background subtraction techniques:

- **MOG2** (Zivkovic, 2004) — Gaussian Mixture Model-based background subtraction
- **KNN** (Zivkovic & van der Heijden, 2006) — K-nearest-neighbors based background subtraction

The goal is to make these techniques easy to apply to real video/webcam input with sensible defaults, shadow removal, morphological noise cleanup, and simple mask/overlay outputs — without needing to hand-roll OpenCV boilerplate every time.

## Why this exists

Most tutorials on background subtraction stop at `cv2.createBackgroundSubtractorMOG2()` with no post-processing, no CLI, and no reusable structure. This toolkit packages the full pipeline (capture → subtract → denoise → mask → overlay/export) into a single, tested, documented interface, so it can be dropped into other CV pipelines (surveillance, video calls, motion analysis, dataset pre-labeling) with one import.

## Features

- MOG2 and KNN backends, switchable via CLI flag or API parameter
- Shadow detection/removal
- Morphological noise cleanup (open/close operations, configurable kernel)
- Works on video files, image sequences, or live webcam input
- Outputs: binary mask, foreground-only image (background masked out), or side-by-side overlay
- Pure OpenCV + NumPy — no GPU or deep learning dependency required

## Installation

```bash
git clone https://github.com/<your-username>/foreground-segmentation.git
cd foreground-segmentation
pip install -r requirements.txt
```

## Quick start

```bash
# Run on a video file, save foreground-only output
python src/foreground_segmentation.py --input video.mp4 --output out.mp4 --method mog2

# Run on webcam, view live
python src/foreground_segmentation.py --input 0 --live --method knn
```

Python API:

```python
from src.foreground_segmentation import ForegroundSegmenter

seg = ForegroundSegmenter(method="mog2", detect_shadows=True)
mask = seg.apply(frame)          # binary mask
fg = seg.extract_foreground(frame)  # foreground-only image
```

## Project structure

```
foreground-segmentation/
├── src/
│   └── foreground_segmentation.py   # core segmenter class + CLI
├── examples/
│   └── example_usage.py             # minimal usage example
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

## Roadmap

- [ ] Add optional deep-learning backend (e.g. RVM or U²-Net) for cases with static/complex backgrounds where classical methods struggle
- [ ] Batch processing mode for image folders
- [ ] Benchmark script against CDNet2014 dataset with IoU/F-measure reporting
- [ ] Docker image for reproducible runs

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## References

- Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," ICPR 2004.
- Z. Zivkovic and F. van der Heijden, "Efficient adaptive density estimation per image pixel for the task of background subtraction," Pattern Recognition Letters, 2006.

## License

MIT — see [LICENSE](LICENSE).
