import cv2
import numpy as np
import pytest

from src.tracking.tracker import TrackerManager

@pytest.fixture
def dummy_config_tracking():
    return {
        'tracking': {
            'tracker_type': "CSRT"
        }
    }

def test_create_and_update_tracker(dummy_config_tracking):
    tracker_manager = TrackerManager(dummy_config_tracking)
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # Create a dummy detection with a bounding box
    detection = {'box': [160, 120, 320, 240]}
    # Update trackers with the detection
    tracks = tracker_manager.update(dummy_frame, [detection])
    # Check that at least one tracker was created and returned
    assert isinstance(tracks, list)
    assert len(tracks) >= 1
    for track in tracks:
        assert 'id' in track
        assert 'box' in track
