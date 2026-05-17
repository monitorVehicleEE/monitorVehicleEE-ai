import supervision as sv
import numpy as np

class VehicleTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.smooth_memory = {}
        self.alpha = 0.6
        self.class_names = {
            0: "motorbike",
            4: "oto",
            5: "xe-tai",
            10: "xe-container"
        }
    
    def smooth_box(self, track_id, box):
        box = np.array(box, dtype=np.float32)
        if track_id not in self.smooth_memory:
            self.smooth_memory[track_id] = box
            return box.astype(int)
        
        previous = self.smooth_memory[track_id]
        smooth = ( self.alpha * box + (1-self.alpha)*previous )
        self.smooth_memory[track_id] = smooth
        return smooth.astype(int)
    
    def update(self, detections):
        if len(detections) == 0:
            return []
        detections_sv = sv.Detections(
            xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections ]),
            confidence=np.array([d[4] for d in detections]),
            # class_id=np.array([0 for _ in detections])
            class_id=np.array([d[5] for d in detections])
        )

        tracked = self.tracker.update_with_detections(detections_sv)
        vehicles = []
        if tracked.tracker_id is None:
            return vehicles

        for box, tid, conf, cls_id in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence, tracked.class_id):
            x1, y1, x2, y2 = map(int, box)
            # smooth
            x1, y1, x2, y2 = self.smooth_box(tid,[x1, y1, x2, y2])
            label = self.class_names.get(int(cls_id), "vehicle")
            vehicles.append((int(tid), x1, y1, x2, y2, float(conf),label))

        return vehicles