from ultralytics import YOLO
from src.config.settings import DEVICE, OCR_CONF_THRESHOLD, OCR_IMG_SIZE

class PlateChar:
    def __init__(
        self,
        model_path,
        device=DEVICE,
        conf=OCR_CONF_THRESHOLD,
        imgsz=OCR_IMG_SIZE
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

        chars = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]

            chars.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "label": label
            })

        # chars = sorted(chars, key=lambda x: x[0])
        # text = "".join([c[1] for c in chars])
        # return text
        return chars
