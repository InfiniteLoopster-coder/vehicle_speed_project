import cv2
import numpy as np

class YOLOv3:
    def __init__(self, config):
        self.weights_path = config['detection']['weights_path']
        self.config_path = config['detection']['config_path']
        self.class_names_path = config['detection']['class_names']
        self.conf_threshold = config['detection']['confidence_threshold']
        self.nms_threshold = config['detection']['nms_threshold']

        # Load the YOLO network
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load class names
        with open(self.class_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get the output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids
