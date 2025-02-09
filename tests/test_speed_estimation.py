# tests/test_speed_estimation.py
import numpy as np
import pytest
from src.speed_estimation.speed_estimator import SpeedEstimator

@pytest.fixture
def dummy_config_speed():
    return {
        'speed_estimation': {
            'smoothing_window': 3,
            'conversion_factor': 3.6  # Conversion factor from m/s to km/h
        }
    }

def test_estimate_speed_insufficient_data(dummy_config_speed):
    estimator = SpeedEstimator(dummy_config_speed)
    # Only one position update for object 0
    estimator.update_position(0, (0, 0))
    speed = estimator.estimate_speed(0, 1.0)
    # With insufficient data, speed should be zero.
    assert speed == 0.0

def test_estimate_speed(dummy_config_speed):
    estimator = SpeedEstimator(dummy_config_speed)
    # Simulate two positions for object 0:
    # From (0,0) to (3,4) meters (distance = 5 meters)
    estimator.update_position(0, (0, 0))
    estimator.update_position(0, (3, 4))
    # dt = 1 second; thus, speed should be 5 m/s converted to km/h: 5 * 3.6 = 18 km/h.
    speed = estimator.estimate_speed(0, 1.0)
    assert abs(speed - 18.0) < 1e-6
