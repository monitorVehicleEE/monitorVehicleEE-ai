import cv2
import numpy as np

class Filter_Plate:                                                       #2000                             #3000
    def __init__(self, method: str = "laplacian", char_threshold: float = 2000.0, plate_threshold: float = 2500.0, area_threshold: float = 50):
        self.method = method
        self.char_threshold = char_threshold
        self.plate_threshold = plate_threshold
        self.area_threshold = area_threshold
        self.min_plate_area_ratio = 0.005
        self.min_plate_width_ratio = 0.02
        self.min_plate_height_ratio = 0.01


    def to_gray(self, img):
        if img is None or img.size == 0:
            return None
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def measure_sharpness(self, img) -> float:
        gray = self.to_gray(img)
        if gray is None:
            return 0.0

        if self.method == "laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())

        else: # có thể thay đổi mrthod khác
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            return float(lap.var())
    
    def is_sharp_plate(self, img) -> bool:
        s = self.measure_sharpness(img)
        return s >= self.plate_threshold

    def char_sharpness(self, plate_img, char_boxes):
        if plate_img is None or plate_img.size == 0:
            for c in char_boxes:
                c["sharpness"] = 0.0
                c["is_blur"] = True
            return char_boxes
        H, W = plate_img.shape[:2]

        for c in char_boxes:
            x1, y1, x2, y2 = c["bbox"]
            x1 = max(0, min(W - 1, int(x1)))
            x2 = max(0, min(W,     int(x2)))
            y1 = max(0, min(H - 1, int(y1)))
            y2 = max(0, min(H,     int(y2)))

            if x2 <= x1 or y2 <= y1:
                c["sharpness"] = 0.0
                c["is_blur"] = True
                continue
            char_crop = plate_img[y1:y2, x1:x2]
            s = self.measure_sharpness(char_crop)
            c["sharpness"] = s
            c["is_blur"] = (s < self.char_threshold)

        return char_boxes

    def avg_sharpness(self, char_boxes, ignore_blur=True) -> float:
        vals = []
        for c in char_boxes:
            s = c.get("sharpness", None)
            if s is None:
                continue
            if ignore_blur and c.get("is_blur", False):
                continue
            vals.append(s)
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def is_plate_size_valid(self, frame, bbox):
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h

        frame_area = W * H

        # kiểm tra theo tỉ lệ diện tích
        if area < self.min_plate_area_ratio * frame_area:
            return False

        # kiểm tra theo tỉ lệ chiều rộng/chiều cao
        if w < self.min_plate_width_ratio * W:
            return False
        if h < self.min_plate_height_ratio * H:
            return False

        return True
