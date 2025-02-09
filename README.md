# Vehicle Speed Detection and Tracking System

![Vehicle Speed Detection Banner](docs/banner.png)

An advanced, modular, real-time computer vision system designed to detect vehicles (cars, bikes, buses) from video streams, track them across frames, estimate their speeds in real-world units, and provide an interactive visualization. This project combines state-of-the-art deep learning detection (e.g., YOLO), classical tracking algorithms, camera calibration techniques, and perspective transformations to deliver accurate and robust performance under a variety of conditions.

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Main Pipeline](#running-the-main-pipeline)
  - [Calibration](#calibration)
  - [Inference](#inference)
- [Experiments and Trial Runs](#experiments-and-trial-runs)
- [Testing](#testing)
- [Contributing](#contributing)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Real-Time Detection:** Leverages deep learning (YOLOv3) for robust vehicle detection in live video streams.
- **Multi-Object Tracking:** Uses OpenCV-based trackers (CSRT, KCF, MIL) to maintain vehicle identities across frames.
- **Camera Calibration & Perspective Transformation:** Calibrate the camera using chessboard images and compute homography to convert image coordinates to real-world measurements.
- **Speed Estimation:** Calculates vehicle speed by analyzing real-world displacement over time, with built-in smoothing to reduce noise.
- **Visualization:** Overlays bounding boxes, object IDs, and speed data on video streams; supports both live display and video recording.
- **Extensive Logging:** Built-in logging for debugging, performance monitoring, and error tracking.
- **Modular and Scalable:** Designed for ease of maintenance, testing, and future extensions.

---

## Folder Structure

vehicle_speed_project/
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Python dependencies (e.g., OpenCV, NumPy, TensorFlow/PyTorch)
├── setup.py                    # Packaging script if you decide to package your project
├── LICENSE                     # Licensing information
├── docs/                       # Documentation (architecture, API docs, design decisions)
│   ├── architecture.md
│   └── calibration_guide.md
├── data/                       # Data related to the project
│   ├── raw/                    # Raw videos/images, calibration images, etc.
│   └── processed/              # Processed data for analysis and debugging
├── experiments/                # Experiments and trial runs
│   ├── calibration_experiments/
│   ├── detection_experiments/
│   └── tracking_experiments/
├── notebooks/                  # Jupyter notebooks for exploratory analysis and tutorials
│   ├── exploratory_analysis.ipynb
│   └── calibration_tutorial.ipynb
├── src/                        # Source code for the project
│   ├── __init__.py
│   ├── config/                 # Configuration files and scripts
│   │   ├── __init__.py
│   │   └── config.yaml         # Centralized configuration file for hyperparameters, file paths, etc.
│   ├── calibration/            # Camera calibration and perspective transformation
│   │   ├── __init__.py
│   │   └── calibration.py      # Functions and classes for camera calibration
│   ├── detection/              # Object detection module
│   │   ├── __init__.py
│   │   ├── detector.py         # Detector class that wraps model loading and inference
│   │   └── models/             # Model files and utilities (if needed)
│   │       ├── __init__.py
│   │       └── yolov3.py       # Example model integration
│   ├── tracking/               # Object tracking module
│   │   ├── __init__.py
│   │   └── tracker.py          # Tracker class and helper functions for tracking objects
│   ├── speed_estimation/       # Speed calculation logic
│   │   ├── __init__.py
│   │   └── speed_estimator.py  # Functions to compute speed using displacement and time differences
│   ├── visualization/          # Visualization and video overlay utilities
│   │   ├── __init__.py
│   │   └── visualizer.py       # Visualization functions to overlay bounding boxes, speed data, etc.
│   ├── logging/                # Logging utilities
│   │   ├── __init__.py
│   │   └── logger.py           # Custom logger configuration and log helper functions
│   ├── utils/                  # Miscellaneous helper functions and utilities
│   │   ├── __init__.py
│   │   └── utils.py
│   └── main.py                 # Main script to initialize and run the full pipeline
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_calibration.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   ├── test_speed_estimation.py
│   └── test_visualization.py
└── scripts/                    # Utility scripts (e.g., for training models or calibration)
    ├── calibrate_camera.py
    └── run_inference.py


Usage
Running the Main Pipeline
To start the full vehicle speed detection system (using your default camera or a video stream), run:

python src/main.py

Calibration
Before running the main pipeline, ensure that your camera is calibrated. Use the provided script:

python scripts/calibrate_camera.py --config src/config/config.yaml

Inference
To run inference on a video or image file, use:

python scripts/run_inference.py --input path/to/your/video.mp4 --config src/config/config.yaml


Experiments and Trial Runs
The experiments/ folder contains dedicated subfolders for experimenting with:

Calibration Experiments:
Test various calibration methods, analyze reprojection errors, and refine homography calculations.

Detection Experiments:
Compare different object detection models and hyperparameter settings using sample images and videos.

Tracking Experiments:
Evaluate and benchmark various tracking algorithms (e.g., CSRT, KCF, MIL) under different conditions.
