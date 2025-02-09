# tests/test_visualization.py
import cv2
import numpy as np
import pytest

from src.visualization.visualizer import Visualizer

@pytest.fixture
def dummy_config_visualization():
    return {
        'visualization': {
            'window_name': "Test Window"
        }
    }

def test_draw_detections(dummy_config_visualization):
    visualizer = Visualizer(dummy_config_visualization)
    # Create a dummy frame (black image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Dummy tracks: list of dictionaries with 'id' and 'box'
    tracks = [
        {'id': 0, 'box': [100, 100, 50, 50]},
        {'id': 1, 'box': [200, 200, 60, 60]}
    ]
    speeds = {
        0: 50.0,
        1: 30.0
    }
    output_frame = visualizer.draw_detections(dummy_frame.copy(), tracks, speeds)
    # The output frame should not be None and must have the same dimensions as the input.
    assert output_frame is not None
    assert output_frame.shape == dummy_frame.shape

def test_show_frame(dummy_config_visualization, monkeypatch):
    visualizer = Visualizer(dummy_config_visualization)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Monkey-patch cv2.imshow to record that it was called.
    called = []

    def fake_imshow(winname, frame):
        called.append(winname)

    monkeypatch.setattr(cv2, 'imshow', fake_imshow)
    visualizer.show_frame(dummy_frame)
    assert dummy_config_visualization['visualization']['window_name'] in called
