from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path, video_source, tracker_cfg = "bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.results_stream = self.model.track(
            source  = video_source,
            tracker = tracker_cfg,
            persist = True,
            stream  =  True
        )

    def next_frame(self):
        result = next(self.results_stream, None)
        if result is None:
            return None, []

        frame = result.orig_img
        vehicles = []

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            bboxes = result.boxes.xyxy.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            # nếu cần bạn có thể lấy từ box.conf
            for tid, bbox, cid, conf  in zip(track_ids, bboxes, class_ids, confs):
                x1, y1, x2, y2 = bbox
                label = self.model.names[cid]
                vehicles.append((tid, x1, y1, x2, y2, float(conf), label))
                
        return frame, vehicles
