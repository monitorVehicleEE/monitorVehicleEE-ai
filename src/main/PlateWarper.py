import cv2
import numpy as np

class PlateWarper:
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

    def warp(self, image, points, expand_scale=1.2, target_ratio=4.0):
        if not PlateWarper.is_valid_plate(points):
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
        wraped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        plate_for_char = cv2.cvtColor(wraped_gray, cv2.COLOR_GRAY2BGR)


        return plate_for_char

        


