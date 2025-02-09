import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.calibration.calibration import CameraCalibrator

# Define a dummy configuration for experiments
config = {
    'calibration': {
        'chessboard_size': [9, 6],
        'calibration_images_dir': "data/raw/calibration",  # Directory with calibration images
        'save_file': "data/processed/calibration_exp.yaml",
        'homography_src_points': [[100, 200], [540, 200], [540, 380], [100, 380]],
        'homography_dst_points': [[0, 0], [4, 0], [4, 3], [0, 3]]
    }
}

# Initialize the calibrator
calibrator = CameraCalibrator(config)

# Option 1: Visualize chessboard detection on calibration images.
image_paths = glob.glob("data/raw/calibration/*.jpg")
if not image_paths:
    print("No calibration images found.")
else:
    reprojection_errors = []
    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, tuple(config['calibration']['chessboard_size']), None)
        if ret:
            # Refine corner detection
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            # Draw and display detected corners
            cv2.drawChessboardCorners(img, tuple(config['calibration']['chessboard_size']), corners2, ret)
            cv2.imshow("Detected Corners", img)
            cv2.waitKey(100)
            # Optionally compute reprojection error for this image.
            # (You could run calibrator.calibrate_camera() on subsets and then compute the error.)
            # For demonstration, we just add a dummy error value.
            reprojection_errors.append(np.random.rand())
    cv2.destroyAllWindows()

    # Plot reprojection errors
    plt.figure(figsize=(8, 4))
    plt.hist(reprojection_errors, bins=10, edgecolor='k')
    plt.title("Reprojection Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.show()

# Option 2: Run full calibration and compute homography.
if calibrator.calibrate_camera():
    print("Calibration successful.")
    print("Camera Matrix:", calibrator.camera_matrix)
    print("Distortion Coefficients:", calibrator.dist_coeffs)
    
    H = calibrator.compute_homography(config['calibration']['homography_src_points'],
                                      config['calibration']['homography_dst_points'])
    print("Homography Matrix:", H)
else:
    print("Calibration failed. Please check your images or configuration.")