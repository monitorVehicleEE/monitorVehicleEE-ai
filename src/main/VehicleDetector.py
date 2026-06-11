from ultralytics import YOLO
from src.config.settings import DEVICE, VEHICLE_CONF_THRESHOLD

class VehicleDetector:
    def __init__(self, model_path, device=DEVICE, conf=VEHICLE_CONF_THRESHOLD):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.conf,
            device=self.device,
            verbose=False
        )[0]
        vehicles = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_vehicle = int(box.cls[0]) 
            conf = float(box.conf[0])
            label = self.model.names[cls_vehicle]  

            vehicles.append((x1, y1, x2, y2,conf,cls_vehicle,label))
            # print(vehicles)

        return vehicles
