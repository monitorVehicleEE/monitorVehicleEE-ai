import os
import cv2
from pathlib import Path
from PIL import Image, ImageDraw


# image_dir = "./dataset/vehicle/motocycle_raw/images/train/"
# label_dir = "./dataset/vehicle/motocycle_raw/labels/train/"
# output_img_dir = "./dataset/vehicle/motorbike_masked/"

src_root    = Path("dataset/vehicle/motocycle_raw")
target_root = Path("dataset/vehicle/motorbike_masked")

for split in ["train","val"]:
    src_images_root = src_root / "images" / split
    src_labels_root = src_root / "labels" / split

    # duyệt file .jpg và jpeg trong các batch
    image_paths = list(src_images_root.rglob("*.jpg")) + \
                  list(src_images_root.rglob("*.jpeg"))   

    # print(f"{split}: found {len(image_paths)} images")

    for image_path in image_paths:
        relative_image_path = image_path.relative_to(src_images_root)
        # cắt còn batch_i/abc.jpg

        # giữ batch đích cho ảnh
        target_image_path = target_root / "images" / split / relative_image_path
        target_image_path.parent.mkdir(parents=True, exist_ok=True)

        # đường dẫn label tương ứng, giữ batch
        src_labels_path = src_labels_root / relative_image_path.with_suffix(".txt")
        target_label_path = target_root / "labels" / split / relative_image_path.with_suffix(".txt")
        target_label_path.parent.mkdir(parents=True, exist_ok=True)

        # mở ảnh
        image = Image.open(image_path).convert("RGB")
        width,height = image.size
        drawer = ImageDraw.Draw(image)

        lines_to_keep = []

        if src_labels_path.exists():
            with open(src_labels_path, "r") as f:
                for line in f.read().strip().splitlines(): # đọc thành 1 line -> cắt trắng \n cuối-> chia thành mảng ( đọc file -> tách từng dòng -> đọc từng dòng)
                    if not line:
                        continue
                    parts = line.split() # tách dòng thành list "1 0.5 0.5 0.2 0.2" -> ['1', '0.5', '0.5', '0.2', '0.2']
                    class_id = int(parts[0]) # lấy class
                    x_center, y_center, box_w, box_h = map(float, parts[1:])

                    # YOLO (0–1) → pixel
                    x_center *= width  # tọa độ trên ảnh
                    y_center *= height
                    box_w *= width     # kích thước thật 
                    box_h *= height
                        # vẽ bounding box
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    if class_id == 1:
                        # class 1 (xoa) → fill trắng
                        drawer.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                    else:
                        # giữ lại các class khác
                        lines_to_keep.append(line)
            
        image.save(target_image_path)
        
        # lưu label đã lọc (không còn class 1), vẫn trong đúng batch
        if lines_to_keep:
            with open(target_label_path, "w") as f:
                f.write("\n".join(lines_to_keep))
        else:
            open(target_label_path, "w").close()