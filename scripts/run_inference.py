#!/usr/bin/env python
"""
Script: run_inference.py
Description: Runs inference on an input video or image using the vehicle detection,
             tracking, and visualization modules. Calibration data is loaded and applied
             to undistort frames. The processed output is displayed and can be saved to a file.
Usage:
    python run_inference.py --input <input_file> [--output <output_file>] --config src/config/config.yaml
"""

import argparse
import yaml
import os
import cv2
from src.calibration.calibration import CameraCalibrator
from src.detection.detector import Detector
from src.tracking.tracker import TrackerManager
from src.visualization.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a video or image file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input video or image file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(Optional) Path to save the output video file.",
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

    # Initialize and load calibration parameters
    calibrator = CameraCalibrator(config)
    if not calibrator.load_calibration():
        print("Calibration file not found or invalid. Please run calibrate_camera.py first.")
        return

    # Compute homography matrix from preset points (if used in your pipeline)
    src_points = config['calibration']['homography_src_points']
    dst_points = config['calibration']['homography_dst_points']
    calibrator.compute_homography(src_points, dst_points)

    # Initialize Detector, Tracker, and Visualizer
    detector = Detector(config)
    tracker_manager = TrackerManager(config)
    visualizer = Visualizer(config)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    # Process a video file
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(input_path)
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply undistortion using calibration data
            frame = calibrator.undistort(frame)

            # Run detection on the current frame
            detections = detector.detect(frame)

            # Update trackers with new detections
            tracks = tracker_manager.update(frame, detections)

            # (Optional) You can extend this script to also estimate speeds.
            # For simplicity, we pass an empty dictionary for speeds.
            output_frame = visualizer.draw_detections(frame, tracks, speeds={})

            cv2.imshow("Inference", output_frame)
            if writer:
                writer.write(output_frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    else:
        # Process a single image file
        frame = cv2.imread(input_path)
        if frame is None:
            print("Failed to read the input image.")
            return

        frame = calibrator.undistort(frame)
        detections = detector.detect(frame)
        tracks = tracker_manager.update(frame, detections)
        output_frame = visualizer.draw_detections(frame, tracks, speeds={})

        cv2.imshow("Inference", output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
