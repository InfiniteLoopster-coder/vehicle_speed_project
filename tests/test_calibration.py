# tests/test_calibration.py
import numpy as np
import cv2
import tempfile
import os
import yaml
import pytest

from src.calibration.calibration import CameraCalibrator

@pytest.fixture
def dummy_config():
    return {
        'calibration': {
            'chessboard_size': [9, 6],
            'calibration_images_dir': 'tests/dummy_calib_images',  # Dummy directory for testing
            'save_file': tempfile.mktemp(suffix='.yaml'),
            'homography_src_points': [[100, 200], [540, 200], [540, 380], [100, 380]],
            'homography_dst_points': [[0, 0], [4, 0], [4, 3], [0, 3]]
        }
    }

def test_compute_homography(dummy_config):
    calibrator = CameraCalibrator(dummy_config)
    src_points = dummy_config['calibration']['homography_src_points']
    dst_points = dummy_config['calibration']['homography_dst_points']
    H = calibrator.compute_homography(src_points, dst_points)
    assert H.shape == (3, 3)
    # Test transformation: convert a sample point
    sample_point = [(320, 290)]
    transformed = calibrator.transform_points(sample_point)
    assert isinstance(transformed, list)
    assert len(transformed) == 1

def test_undistort(dummy_config):
    calibrator = CameraCalibrator(dummy_config)
    # Set a dummy camera matrix and zero distortion coefficients
    calibrator.camera_matrix = np.array([[1000, 0, 320],
                                           [0, 1000, 240],
                                           [0, 0, 1]], dtype=np.float32)
    calibrator.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    undistorted = calibrator.undistort(dummy_img)
    # The undistorted image should have the same shape as the original
    assert undistorted.shape == dummy_img.shape

def test_load_and_save_calibration(dummy_config):
    # Write a dummy calibration file and test that load_calibration retrieves it correctly.
    dummy_data = {
        'camera_matrix': [[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]],
        'dist_coeffs': [[0], [0], [0], [0], [0]]
    }
    with open(dummy_config['calibration']['save_file'], 'w') as f:
        yaml.dump(dummy_data, f)
    calibrator = CameraCalibrator(dummy_config)
    loaded = calibrator.load_calibration()
    assert loaded is True
    np.testing.assert_array_almost_equal(
        calibrator.camera_matrix, np.array(dummy_data['camera_matrix'])
    )
