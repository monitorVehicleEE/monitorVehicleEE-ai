from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, conf=0.4)[0]

        plates = []

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        
            if results.keypoints is not None:
                kpts = results.keypoints.xy[i].cpu().numpy()
                if len(kpts) >= 4:
                    pts = kpts[:4]
                else:
                    continue
            else:
                pts = None

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "points": pts
            })
        return plates