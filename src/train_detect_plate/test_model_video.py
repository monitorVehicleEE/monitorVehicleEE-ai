from ultralytics import YOLO
import cv2
import os

model = YOLO("./runs/pose/runs_detect_plate/yl11s_dp_ver3/weights/best.pt")

video_path = "./dataset/vehicle/test_videos/25.mp4"
cap = cv2.VideoCapture(video_path)

# lấy thông tin video
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_dir = "./runs/pose/runs/pose/video_results/"
os.makedirs(output_dir, exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = os.path.join(output_dir, f"{video_name}_result_2.mp4")

# lưu video output
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)

    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        kpts  = r.keypoints.xy.cpu().numpy()

        for box, kp in zip(boxes, kpts):
            x1, y1, x2, y2 = map(int, box)

            # bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

            # keypoints
            for (x,y) in kp:
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), -1)

            # polygon (4 cạnh)
            pts = kp.astype(int)
            cv2.polylines(frame, [pts], True, (255,0,0), 1)

    # hiển thị
    cv2.imshow("video", frame)

    # lưu
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
out.release()
cv2.destroyAllWindows()
