from ultralytics import YOLO

class PlateChar:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, conf=0.3)[0]

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