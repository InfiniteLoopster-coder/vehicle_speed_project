import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.tracking.tracker import TrackerManager
import yaml

# Load configuration for tracking experiments.
config_path = "src/config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# You might want to test different tracker types. Here we loop over a list.
tracker_types = ["CSRT", "KCF", "MIL"]
results = {}

# Path to a sample video for tracking experiments.
video_path = "experiments/tracking_experiments/sample_video.mp4"

for tracker_type in tracker_types:
    print(f"Evaluating tracker: {tracker_type}")
    config['tracking']['tracker_type'] = tracker_type
    tracker_manager = TrackerManager(config)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    successful_tracks = 0
    trajectories = {}  # object_id -> list of centers

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # For demonstration, suppose we simulate detections with a fixed bounding box.
        # In a real experiment, you would run a detector first.
        dummy_detection = {'box': [100, 100, 100, 100]}
        tracks = tracker_manager.update(frame, [dummy_detection])
        for track in tracks:
            obj_id = track['id']
            x, y, w, h = track['box']
            center = (x + w // 2, y + h // 2)
            if obj_id not in trajectories:
                trajectories[obj_id] = []
            trajectories[obj_id].append(center)
            successful_tracks += 1
        
        # Optionally, draw current tracks on the frame.
        for track in tracks:
            x, y, w, h = track['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow(f"Tracker {tracker_type}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # Store summary metrics for this tracker type.
    results[tracker_type] = {
        'frames_processed': frame_count,
        'successful_tracks': successful_tracks,
        'trajectories': trajectories  # optionally, you can compute additional metrics
    }

# Print out summary metrics for each tracker type.
for tracker_type, data in results.items():
    print(f"Tracker: {tracker_type}")
    print(f"  Frames Processed: {data['frames_processed']}")
    print(f"  Total Successful Tracking Updates: {data['successful_tracks']}")
    # Optionally, plot trajectories.
    for obj_id, traj in data['trajectories'].items():
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Obj {obj_id}")
    plt.title(f"Trajectories for Tracker: {tracker_type}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()