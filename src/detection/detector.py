import cv2
import numpy as np
from detection.models.yolov3 import YOLOv3

class Detector:
    def __init__(self, config):
        self.conf_threshold = config['detection']['confidence_threshold']
        self.nms_threshold = config['detection']['nms_threshold']
        self.model = YOLOv3(config)

    def detect(self, frame):
        # Run detection on the frame using the YOLOv3 model
        boxes, confidences, class_ids = self.model.detect(frame)
        # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i]
                })
        return detections
