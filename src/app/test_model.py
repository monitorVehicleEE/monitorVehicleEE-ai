import cv2
import time
from ultralytics import YOLO

# =========================
# CONFIG
# =========================

MODEL_PATH = "./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt"
# MODEL_PATH = "best.pt"

VIDEO_SOURCE = "./dataset/vehicle/videos/27.mp4"

IMG_SIZE = 640
CONF = 0.5

# =========================
# LOAD MODEL
# =========================

print("Loading model...")
model = YOLO(MODEL_PATH)

print("Model loaded!")

# =========================
# WARMUP
# =========================

# print("Warming up model...")

# dummy = cv2.imread("test.jpg")

# if dummy is not None:
#     model.predict(dummy, imgsz=IMG_SIZE)

# print("Warmup done!")

# =========================
# VIDEO CAPTURE
# =========================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Cannot open camera/video")
    exit()

prev_time = time.time()

# =========================
# LOOP
# =========================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Inference
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        verbose=False
    )

    # Draw result
    annotated_frame = results[0].plot()

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show
    cv2.imshow("YOLO TensorRT Test", annotated_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print("Done!")