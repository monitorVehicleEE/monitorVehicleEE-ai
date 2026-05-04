import os
import cv2
import numpy as np
from pathlib import Path

TARGET_CLASS = 0
IMG_SIZE     = (320, 96)

src_root = Path("./dataset/plate-detect/data_raw")
img_root = src_root / "images"
label_root = src_root / "labels"

out_root = Path("./dataset/data_warped")
out_root.mkdir(exist_ok=True)

IMG_TYPE = [".jpg", ".jpeg", ".png"]

def find_image(img_dir, stem):
    for ext in IMG_TYPE:
        path = img_dir / (stem + ext)
        if path.exists():
            return path
    return None

def yolo_to_pixel(keypoints, img_w, img_h): # chuyển đổi pose về pixel cho opencv
    points = []
    for i in range(4):
        # giả sử ở đây
        # kps = [
        #     0.6, 0.3, 2,
        #     0.7, 0.3, 2,
        #     0.7, 0.4, 2,
        #     0.6, 0.4, 2
        # ]
        # i = 0: 
        # x = kps[0] = 0.6
        # y = kps[1] = 0.3
        # i = 1:
        # x = kps[3] = 0.7
        # y = kps[4] = 0.3
        # nếu không sẽ lấy nhầm y chung với x
        x = keypoints[i*3] * img_w   # mỗi kpt 3 phần tử x,y,visible nên mỗi lần nhảy i phải nhảy 3 phần tử
        y = keypoints[i*3 + 1] * img_h
        points.append([x,y])

    return np.array(points, dtype="float32")

def order_point(points): # sắp xếp lại thứ tự 4 điểm
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

def expand_points(points, scale=1.2):
    center = np.mean(points, axis=0)
    expanded = center + (points - center) * scale
    return expanded.astype(np.float32)

def is_valid_plate(points): # kiểm tra có nên dùng wrap không
    area = cv2.contourArea(points.astype(np.int32)) #tính diện tích tứ giác
    if area < 20:
        return False # loại bỏ các kpt sai và biển số quá nhỏ, 4 điểm gần như trùng

    if len(np.unique(points, axis=0)) < 4: # kiểm tra trùng điểm
        return False

    return True

def warp_plate(image, points):
    # chuẩn hóa thứ tự 4 điểm
    rect = order_point(points)
    rect = expand_points(rect, scale=1.2)
    (tl,tr,br,bl)  = rect
    
    widthA = np.linalg.norm(br - bl) # chiều dài của vector theo sqrt(x^2+y^2)
    widthB  = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 5 or maxH < 5:
        return None

    # tạo ảnh đích
    dst = np.array([
        [0,0],
        [maxW-1,0],
        [maxW-1, maxH-1],
        [0, maxH-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst) # tìm ma trận biển đổi tứ giác bất kì thành hcn
    warped = cv2.warpPerspective(image, M, (maxW, maxH)) # wrap ảnh

    return warped


label_files = list(label_root.rglob("*.txt"))
for label_path in label_files:
    # giữ cấu trúc thư mục map sang nhau
    rel_path = label_path.relative_to(label_root)
    img_dir = img_root / rel_path.parent

    img_path = find_image(img_dir, label_path.stem)

    if img_path is None:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2] # convert yolo sang pixel
    with open(label_path, "r") as f:
        lines = f.readlines()

    out_dir = out_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls = int(parts[0])

        if cls == 2:
            continue

        keypoints = parts[5:]

        if len(keypoints) < 12: # kiểm tra đủ 4 x 3 keypoint
            continue

        points = yolo_to_pixel(keypoints, w, h)

        if not is_valid_plate(points):
            continue

        warped = warp_plate(img, points)

        if warped is None:
            continue

        warped = cv2.resize(warped, IMG_SIZE)
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        save_name = f"{label_path.stem}_{count}.jpg"
        save_path = out_dir / save_name

        cv2.imwrite(str(save_path), warped)

        count += 1

print("DONE ALL DATASET!")









    



    

