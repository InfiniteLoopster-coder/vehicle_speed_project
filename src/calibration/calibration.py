import cv2
import numpy as np
import glob
import yaml
import os

class CameraCalibrator:
    def __init__(self, config):
        self.chessboard_size = tuple(config['calibration']['chessboard_size'])
        self.calib_images_dir = config['calibration']['calibration_images_dir']
        self.save_file = config['calibration']['save_file']
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography_matrix = None

    def calibrate_camera(self):
        # Prepare object points based on the real-world coordinates
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                                0:self.chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane

        # Get list of calibration images
        images = glob.glob(os.path.join(self.calib_images_dir, '*.jpg'))
        if not images:
            print("No calibration images found in", self.calib_images_dir)
            return False

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(100)
        cv2.destroyAllWindows()

        ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        # Save calibration parameters to file
        calib_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist()
        }
        with open(self.save_file, 'w') as f:
            yaml.dump(calib_data, f)
        print("Calibration successful. Data saved to", self.save_file)
        return True

    def load_calibration(self):
        if not os.path.exists(self.save_file):
            print("Calibration file not found:", self.save_file)
            return False
        with open(self.save_file, 'r') as f:
            calib_data = yaml.safe_load(f)
            self.camera_matrix = np.array(calib_data['camera_matrix'])
            self.dist_coeffs = np.array(calib_data['dist_coeffs'])
        return True

    def undistort(self, img):
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Calibration parameters not loaded.")
            return img
        return cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

    def compute_homography(self, src_points, dst_points):
        """
        Compute homography matrix from image (src_points) to real-world (dst_points) coordinates.
        """
        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self.homography_matrix, status = cv2.findHomography(src, dst)
        return self.homography_matrix

    def transform_points(self, points):
        """
        Transform a list of (x, y) points from image coordinates to real-world coordinates.
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not computed.")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        return [tuple(pt[0]) for pt in transformed]
