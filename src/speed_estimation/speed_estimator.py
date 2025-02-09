import numpy as np
from collections import deque

class SpeedEstimator:
    def __init__(self, config):
        self.smoothing_window = config['speed_estimation']['smoothing_window']
        self.conversion_factor = config['speed_estimation']['conversion_factor']  # e.g., 3.6 to convert m/s to km/h
        # Store the history of positions for each object (object_id: deque of positions)
        self.positions = {}

    def update_position(self, obj_id, position):
        """
        Update the position history for an object.
        Position should be in real-world coordinates (e.g., meters).
        """
        if obj_id not in self.positions:
            self.positions[obj_id] = deque(maxlen=self.smoothing_window)
        self.positions[obj_id].append(position)

    def estimate_speed(self, obj_id, dt):
        """
        Estimate the speed for a given object using the displacement over time dt.
        Returns speed in km/h.
        """
        if obj_id not in self.positions or len(self.positions[obj_id]) < 2:
            return 0.0
        pts = list(self.positions[obj_id])
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        distance = np.sqrt(dx * dx + dy * dy)  # Distance in meters (assumed)
        speed_mps = distance / dt if dt > 0 else 0.0  # Speed in m/s
        speed = speed_mps * self.conversion_factor  # Convert to km/h if conversion_factor is 3.6
        return speed
