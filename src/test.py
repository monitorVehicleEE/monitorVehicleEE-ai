# import torch
# from ultralytics import YOLO
# model = YOLO("./model/pytorch/vehicle/best.pt") 

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.get_device_name(0))

# x = torch.randn(3, 3).cuda()
# y = torch.randn(3, 3).cuda()
# print((x @ y).device)
# print(next(model.parameters()).device)

# test_trt_engine.py
import tensorrt as trt
print(trt.__version__)

import cv2
import time
from ultralytics import YOLO

# =========================
# LOAD TENSORRT ENGINE
# =========================
model = YOLO("model/pytorch/vehicle/best.engine", task="detect")

# =========================
# VIDEO SOURCE
# =========================
video_path = "./dataset/vehicle/videos/27.mp4"  # đổi thành video của bạn

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không mở được video")
    exit()

# =========================
# FPS
# =========================
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # =========================
    # INFERENCE TRT
    # =========================
    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    # =========================
    # FPS
    # =========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("TensorRT Detection", annotated_frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()