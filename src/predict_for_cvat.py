from ultralytics import YOLO
import os

model = YOLO("./runs/detect/runs_motorbike/yolo11_motorbike_gpu_ver3/weights/best.pt")

input_folder = "./dataset/vehicle/motocycle_raw/images/train/f/"
output_label_folder = "./dataset/vehicle/motocycle_raw/labels/train/f/"

os.makedirs(output_label_folder, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")

total_images = 0
total_labels = 0

for root, _, files in os.walk(input_folder):
    for img_name in files:
        if not img_name.lower().endswith(image_extensions):
            continue

        total_images += 1

        img_path = os.path.join(root, img_name)
        rel_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_label_folder, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        label_path = os.path.join(
            save_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        results = model.predict(
            source=img_path,
            conf=0.1,      # giảm ngưỡng để tăng khả năng phát hiện
            imgsz=640,
            verbose=False
        )[0]

        h, w = results.orig_shape

        with open(label_path, "w") as f:
            if results.boxes is not None and len(results.boxes) > 0:
                total_labels += 1
                for box in results.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Chuyển sang định dạng YOLO
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h

                    f.write(
                        f"{cls} {x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f}\n"
                    )

print(f"Processed {total_images} images.")
print(f"Images with detections: {total_labels}")
print("Auto-labeling completed!")
