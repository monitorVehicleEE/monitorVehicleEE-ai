from ultralytics import YOLO
import numpy as np

from src.config.settings import DEVICE, PLATE_CONF_THRESHOLD, PLATE_IMG_SIZE

class PlateDetector:
    def __init__(
        self,
        model_path,
        device=DEVICE,
        conf=PLATE_CONF_THRESHOLD,
        imgsz=PLATE_IMG_SIZE
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        
    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]

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

    def warmup(self, imgsz=None):
        size = int(imgsz or self.imgsz or PLATE_IMG_SIZE)
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        self.detect(dummy)
