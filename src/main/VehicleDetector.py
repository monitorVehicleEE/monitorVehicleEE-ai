from ultralytics import YOLO

class VehicleDetector:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, conf = 0.4)[0] # 1 frame -> kêt quả của frame đó
        vehicles = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_vehicle = int(box.cls[0]) 
            conf = float(box.conf[0])
            label = self.model.names[cls_vehicle]  

            vehicles.append((x1, y1, x2, y2,conf,cls_vehicle,label))
            # print(vehicles)

        return vehicles