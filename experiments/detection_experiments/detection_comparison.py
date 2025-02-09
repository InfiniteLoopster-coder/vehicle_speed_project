import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.detection.detector import Detector
import yaml

# Load a dummy configuration (or modify the production one) for detection experiments
config_path = "src/config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Optionally, override certain detection parameters for experiments
config['detection']['confidence_threshold'] = 0.5
config['detection']['nms_threshold'] = 0.4

# Initialize the detector
detector = Detector(config)

# Load a set of sample images from an experiments folder (ensure you have some test images)
sample_image_paths = ["experiments/detection_experiments/test1.jpg",
                      "experiments/detection_experiments/test2.jpg"]

results = []

for img_path in sample_image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    detections = detector.detect(img)
    # Draw detections for visualization
    for detection in detections:
        x, y, w, h = detection['box']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Class: {detection['class_id']} ({detection['confidence']:.2f})"
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    results.append(img)

# Display results
for i, res_img in enumerate(results):
    cv2.imshow(f"Detection Result {i}", res_img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the images to an output folder for later analysis.