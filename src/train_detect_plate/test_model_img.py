from ultralytics import YOLO
import cv2

model = YOLO("./runs/pose/runs_detect_plate/yl11s_dp_ver3/weights/best.pt")

input_folder = "test_images"      # folder ảnh input
output_folder = "test_results"    # folder lưu kết quả

os.makedirs(output_folder, exist_ok=True)

image_paths = list(Path(input_folder).glob("*.jpg")) + \
              list(Path(input_folder).glob("*.png")) + \
              list(Path(input_folder).glob("*.jpeg"))

for img_path in image_paths:
    results = model(str(img_path))

    for r in results:
        img = r.orig_img.copy()

        if r.boxes is None or r.keypoints is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        kpts  = r.keypoints.xy.cpu().numpy()

        for box, kp in zip(boxes, kpts):
            x1, y1, x2, y2 = map(int, box)

            # bbox mỏng
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)

            # keypoints
            for (x,y) in kp:
                cv2.circle(img, (int(x), int(y)), 2, (0,0,255), -1)

            # polygon mỏng (giống CVAT)
            pts = kp.astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=(255,0,0), thickness=1)

        # lưu ảnh
        save_path = os.path.join(output_folder, img_path.name)
        cv2.imwrite(save_path, img)

        print(f"Saved: {save_path}")