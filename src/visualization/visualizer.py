import cv2

class Visualizer:
    def __init__(self, config):
        self.window_name = config['visualization']['window_name']

    def draw_detections(self, frame, tracks, speeds):
        """
        Overlay bounding boxes, object IDs, and speed information on the frame.
        - tracks: List of dicts with keys 'id' and 'box'
        - speeds: Dictionary mapping object IDs to speed values
        """
        for track in tracks:
            obj_id = track['id']
            x, y, w, h = track['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"ID: {obj_id}"
            if obj_id in speeds:
                label += f" {speeds[obj_id]:.2f} km/h"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return frame

    def show_frame(self, frame):
        cv2.imshow(self.window_name, frame)
