# tests/test_detection.py
import cv2
import numpy as np
import pytest

from src.detection.detector import Detector

@pytest.fixture
def dummy_config_detection():
    return {
        'detection': {
            'model': "yolov3",
            'weights_path': "tests/dummy.weights",
            'config_path': "tests/dummy.cfg",
            'class_names': "tests/dummy.names",
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }
    }

# Create a dummy YOLOv3 model to override actual inference.
class DummyYOLOv3:
    def __init__(self, config):
        self.conf_threshold = config['detection']['confidence_threshold']
    def detect(self, frame):
        # Return a dummy detection: one box centered in the frame.
        height, width = frame.shape[:2]
        box = [width // 4, height // 4, width // 2, height // 2]
        return [box], [0.9], [0]  # box, confidence, class_id

@pytest.fixture(autouse=True)
def patch_yolov3(monkeypatch, dummy_config_detection):
    from src.detection.models import yolov3
    monkeypatch.setattr(yolov3, 'YOLOv3', DummyYOLOv3)

def test_detector(dummy_config_detection):
    detector = Detector(dummy_config_detection)
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)
    if detections:
        # Each detection should be a dict with keys 'box', 'confidence', and 'class_id'
        assert 'box' in detections[0]
        assert 'confidence' in detections[0]
        assert 'class_id' in detections[0]
