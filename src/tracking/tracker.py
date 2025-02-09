import cv2

class TrackerManager:
    def __init__(self, config):
        self.tracker_type = config['tracking']['tracker_type']
        self.trackers = {}  # Maps object_id to tracker instance
        self.next_id = 0

    def create_tracker(self, frame, box):
        # Instantiate a new tracker based on the configured type
        if self.tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        else:
            tracker = cv2.TrackerMIL_create()
        tracker.init(frame, tuple(box))
        return tracker

    def update(self, frame, detections):
        """
        Update existing trackers and associate them with new detections.
        Returns a list of tracking info dictionaries containing object IDs and bounding boxes.
        """
        updated_tracks = []

        # Update existing trackers
        remove_ids = []
        for obj_id, tracker in self.trackers.items():
            success, box = tracker.update(frame)
            if success:
                updated_tracks.append({'id': obj_id, 'box': list(map(int, box))})
            else:
                remove_ids.append(obj_id)

        # Remove trackers that have failed
        for obj_id in remove_ids:
            self.trackers.pop(obj_id)

        # For each new detection, create a new tracker.
        # (In a more advanced system, you would perform data association to avoid duplicates.)
        for detection in detections:
            box = detection['box']
            tracker = self.create_tracker(frame, box)
            self.trackers[self.next_id] = tracker
            updated_tracks.append({'id': self.next_id, 'box': box})
            self.next_id += 1

        return updated_tracks
