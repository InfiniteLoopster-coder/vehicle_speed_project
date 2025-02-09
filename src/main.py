import cv2
import time
import yaml
import os

from calibration.calibration import CameraCalibrator
from detection.detector import Detector
from tracking.tracker import TrackerManager
from speed_estimation.speed_estimator import SpeedEstimator
from visualization.visualizer import Visualizer
from logging.logger import get_logger

def load_config():
    # Load configuration from the YAML file
    config_path = os.path.join(os.path.dirname(__file__), 'config/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration parameters
    cfg = load_config()

    # Setup logger
    logger = get_logger(cfg)
    logger.info("Starting Vehicle Speed Detection System")

    # Initialize the camera calibrator
    calibrator = CameraCalibrator(cfg)
    if not calibrator.load_calibration():
        logger.info("Calibration data not found. Running calibration...")
        if not calibrator.calibrate_camera():
            logger.error("Calibration failed. Exiting.")
            return

    # Compute the homography matrix using preset points from config
    src_points = cfg['calibration']['homography_src_points']
    dst_points = cfg['calibration']['homography_dst_points']
    homography = calibrator.compute_homography(src_points, dst_points)
    logger.info("Computed Homography Matrix")

    # Initialize detector, tracker, speed estimator, and visualizer
    detector = Detector(cfg)
    logger.info("Detector initialized")
    tracker_manager = TrackerManager(cfg)
    logger.info("Tracker Manager initialized")
    speed_estimator = SpeedEstimator(cfg)
    logger.info("Speed Estimator initialized")
    visualizer = Visualizer(cfg)
    logger.info("Visualizer initialized")

    # Open video capture
    cap = cv2.VideoCapture(cfg['camera']['id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['camera']['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['camera']['frame_height'])
    cap.set(cv2.CAP_PROP_FPS, cfg['camera']['frame_rate'])

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break

        # Undistort the captured frame using calibration data
        frame = calibrator.undistort(frame)

        # Run object detection
        detections = detector.detect(frame)

        # Update trackers based on the current detections
        tracks = tracker_manager.update(frame, detections)

        # Calculate time elapsed
        current_time = time.time()
        dt = current_time - prev_time if (current_time - prev_time) > 0 else 1e-3

        # Estimate speed for each tracked object
        speeds = {}
        for track in tracks:
            x, y, w, h = track['box']
            center = (x + w / 2, y + h / 2)  # Center of bounding box in image coordinates
            # Convert center to real-world coordinates using homography
            real_world_center = calibrator.transform_points([center])[0]
            speed_estimator.update_position(track['id'], real_world_center)
            speed = speed_estimator.estimate_speed(track['id'], dt)
            speeds[track['id']] = speed

        prev_time = current_time

        # Visualize detection, tracking, and speed estimation results on the frame
        output_frame = visualizer.draw_detections(frame, tracks, speeds)
        visualizer.show_frame(output_frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1)
        if key == ord('q'):
            logger.info("Exit signal received. Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
