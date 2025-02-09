#!/usr/bin/env python
"""
Script: calibrate_camera.py
Description: Performs camera calibration using calibration images.
             The calibration parameters (camera matrix and distortion coefficients)
             are saved to a YAML file for later use.
Usage:
    python calibrate_camera.py --config src/config/config.yaml
"""

import argparse
import yaml
import os
from src.calibration.calibration import CameraCalibrator

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run camera calibration using provided calibration images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the configuration YAML file.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create the calibrator object with the configuration settings
    calibrator = CameraCalibrator(config)
    print("Starting camera calibration...")

    # Run the calibration routine.
    # This will display calibration images with detected chessboard corners.
    if calibrator.calibrate_camera():
        print("Camera calibration completed successfully.")
    else:
        print("Camera calibration failed. Check calibration images and configuration.")

if __name__ == "__main__":
    main()
