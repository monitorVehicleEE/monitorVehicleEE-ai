import cv2
import numpy as np

class PlateWarper:
    def __init__(self, sharpness_threshold=100.0, sharpness_method='laplacian'):
        self.sharpness_threshold = sharpness_threshold
        self.sharpness_method = sharpness_method
    @staticmethod
    def order_points(points):
        rect = np.zeros((4,2), dtype="float32") # tạo mảng rỗng 4 hàng 2 cột với 0
        s = points.sum(axis = 1) # cộng x,y cho từng điểm để xác định các góc trong hcn - axis = 1: tính theo hàng   
        #argmin: index của giá trị nhỏ nhất
        # 2 giá trị nhỏ và lớn nhất cho TL và BR
        # đi chéo xuống
        rect[0] = points[np.argmin(s)] #top_left - nhỏ cả x và y 
        rect[2] = points[np.argmax(s)] #bottom_right - lớn cả x và y

        diff = np.diff(points, axis=1)
        # nghiêng trái/phải
        rect[1] = points[np.argmin(diff)] # 
        rect[3] = points[np.argmax(diff)] # 

        # để tránh các trường hợp khó thì ta sẽ giới hạn về góc độ nghiêng để tránh trường hợp đó xảy ra
        return rect
    @staticmethod
    def expand_points(points, scale=1.1):
        center = np.mean(points, axis=0)
        expanded = center + (points - center) * scale
        return expanded.astype(np.float32)

    @staticmethod
    def is_valid_plate(points, min_area=50): # kiểm tra có nên dùng wrap không
        area = cv2.contourArea(points.astype(np.int32)) #tính diện tích tứ giác
        if area < 50:
            return False # loại bỏ các kpt sai và biển số quá nhỏ, 4 điểm gần như trùng

        if len(np.unique(points, axis=0)) < 4: # kiểm tra trùng điểm
            return False

        return True

    @staticmethod
    def calculate_sharpness(image): #Tính độ nét của ảnh
        if image is None or image.size == 0:
            return 0.0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # threshold để nổi ký tự
        th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,10   )
        
        # tìm contour ký tự
        contours, _ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        char_scores = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            area = w * h

            # lọc noise
            if area < 30:
                continue

            # tỷ lệ giống ký tự
            ratio = h / max(w, 1)

            if ratio < 1.0 or ratio > 5.0:
                continue
            # crop char
            char_crop = gray[y:y+h, x:x+w]    

            score = cv2.Laplacian(char_crop,cv2.CV_64F).var()        
            char_scores.append(score)
        if len(char_scores) == 0:
            return 0.0

        return np.mean(char_scores)    

    def warp(self, image, points, expand_scale=1.15, target_ratio=4.0, return_sharpness = False):
        if not PlateWarper.is_valid_plate(points):
            if return_sharpness:
                return None, 0.0, False
            return None
        # ORDER POINT
        rect = PlateWarper.order_points(points)
        rect = PlateWarper.expand_points(rect, scale=expand_scale)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = int(max(heightA, heightB))

        if maxW < 10 or maxH < 10:
            return None

        # ratio = maxW / maxH
        # if ratio < 1.5 or ratio > 8.0:
        #     return None

        # if ratio > target_ratio:
        #     maxH = int(maxW / target_ratio)
        # else:
        #     maxW = int(maxH * target_ratio)

        # TẠO 4 ĐIỂM ĐÍCH
        dst = np.array([
            [0, 0],
            [maxW - 1, 0],
            [maxW - 1, maxH - 1],
            [0, maxH - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image,M,(maxW, maxH))

        # sharpness_score = self.calculate_sharpness(warped)
        # is_sharp_enough = sharpness_score >= self.sharpness_threshold
        # global_sharpness = self.calculate_sharpness(
        #     warped,
        #     method=self.sharpness_method
        # )
        # char_sharpness = self.calculate_char_sharpness(warped)
        # # combine score
        # final_score = (
        #     global_sharpness * 0.4 +
        #     char_sharpness * 0.6
        # )
        # is_sharp_enough = final_score >= self.sharpness_threshold

        wraped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        plate_for_char = cv2.cvtColor(wraped_gray, cv2.COLOR_GRAY2BGR)

        # if return_sharpness:
        #     return (
        #         plate_for_char,
        #         {
        #             "global": global_sharpness,
        #             "char": char_sharpness,
        #             "final": final_score
        #         },
        #         is_sharp_enough
        #     )
        # if not is_sharp_enough:
        #     return None

        return plate_for_char

        


