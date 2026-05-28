from threading import Thread, Lock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Request

from src.main.MainPipeline import MainPipeline
from src.main.CameraRunner import CameraRunner
from src.main.VehicleTracker import VehicleTracker
from src.main.PlateDetector import PlateDetector
from src.main.PlateChar import PlateChar
from src.main.VehicleDetector import VehicleDetector


import cv2
import time

STREAM_TIMING_LOG = False
STREAM_JPEG_QUALITY = 75

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# LOCK
# =========================================================

camera_lock = Lock()

# =========================================================
# LOAD YOLO MODEL 1 LẦN
# =========================================================

detector_vehicle = VehicleDetector("./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt", device=0)

detector_plate = PlateDetector("./runs/pose/runs_detect_plate/yl11s_dp_ver6/weights/best.pt", device=0)

detector_char = PlateChar("./runs/detect/runs_read_plate/yolo11s_read_plate_v6/weights/best.pt", device=0)

# =========================================================
# CAMERA STORE
# =========================================================

camera_runners = {}

# =========================================================
# CAMERA SOURCE MAP
# =========================================================

CAMERA_SOURCES = {
    "27": "./dataset/vehicle/videos/27.mp4",
    "entry_cam": "./dataset/vehicle/videos/27.mp4",
    "exit_cam": "./dataset/vehicle/videos/27.mp4",
}

# =========================================================
# REMOVE CAMERA
# =========================================================

def remove_camera(cam_id):

    with camera_lock:

        if cam_id in camera_runners:

            del camera_runners[cam_id]

            print(f"[INFO] Removed camera: {cam_id}")

# =========================================================
# MJPEG GENERATOR
# =========================================================
def resize_keep_ratio(frame, target_width):
    h, w = frame.shape[:2]

    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h))

def generate_frames(runner, width=640):

    try:
        while runner.running:

            frame = runner.latest_frame

            if frame is None:
                time.sleep(0.01)
                continue
            
            original_h, original_w = frame.shape[:2]
            # print(f"[ORIGINAL] {original_w}x{original_h}")


            # resize giữ tỉ lệ
            if width is not None:
                t0 = time.perf_counter()

            frame = resize_keep_ratio(frame, width)

            resize_time = time.perf_counter() - t0

            if STREAM_TIMING_LOG:
                print(f"[RESIZE] {resize_time*1000:.1f} ms")

            resized_h, resized_w = frame.shape[:2]

            # print(f"[RESIZED FRAME] {frame.shape[1]}x{frame.shape[0]}")

            t0 = time.perf_counter()

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
            )

            jpeg_time = time.perf_counter() - t0

            if STREAM_TIMING_LOG:
                print(f"[JPEG] {jpeg_time*1000:.1f} ms")

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes +
                b"\r\n"
            )

            # time.sleep(0.03)
            time.sleep(getattr(runner, "frame_interval", 1/15))

    except Exception as e:
        print("[STREAM ERROR]", e)

# =========================================================
# STREAM
# =========================================================

@app.get("/stream/{cam_id}")
def stream(cam_id: str, request: Request):

    try:
        width = int(request.query_params.get("w", 640))
    except:
        width = 640

    runner = camera_runners.get(cam_id)

    if runner is None:
        return {"error": "camera not found"}

    return StreamingResponse(
        generate_frames(runner, width),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# =========================================================
# START STREAM
# =========================================================
    
@app.post("/start-stream/{cam_id}")
def start_stream(cam_id: str):

    with camera_lock:

        # camera đang chạy
        if cam_id in camera_runners:

            runner = camera_runners[cam_id]

            if runner.running:

                return {
                    "status": "already_running",
                    "cam_id": cam_id
                }

        # source
        video_source = CAMERA_SOURCES.get(cam_id)

        if video_source is None:

            return {
                "error": f"camera source not found: {cam_id}"
            }

        # tracker riêng cho từng camera
        tracker_vehicle = VehicleTracker()

        # pipeline riêng cho từng camera
        pipeline = MainPipeline(
            vehicle_detector=detector_vehicle,
            vehicle_tracker=tracker_vehicle,
            plate_detector=detector_plate,
            char_detector=detector_char
        )

        # runner
        runner = CameraRunner(
            cam_id=cam_id,
            video_source=video_source,
            pipeline=pipeline,
            save_dir="./dataset/output_test/",
            show=False,
            drop_late_frames=True,
            max_frame_skip=1)

        runner.on_finish = remove_camera

        runner.setup()

        # thread detect
        t = Thread(
            target=runner.run_loop,
            daemon=True
        )

        t.start()

        camera_runners[cam_id] = runner

        print(f"[INFO] Started camera: {cam_id}")

        return {
            "status": "started",
            "cam_id": cam_id
        }

# =========================================================
# STOP STREAM
# =========================================================

@app.post("/stop-stream/{cam_id}")
def stop_stream(cam_id: str):

    with camera_lock:

        runner = camera_runners.get(cam_id)

        if runner is None:

            return {
                "error": "camera not found"
            }

        runner.stop()

        runner.finalize()

        del camera_runners[cam_id]

        print(f"[INFO] Stopped camera: {cam_id}")

        return {
            "status": "stopped",
            "cam_id": cam_id
        }

# =========================================================
# CAMERA STATUS
# =========================================================

@app.get("/camera-status/{cam_id}")
def camera_status(cam_id: str):

    runner = camera_runners.get(cam_id)

    if runner is None:

        return {
            "running": False
        }

    return {
        "running": runner.running
    }

# =========================================================
# LIST CAMERAS
# =========================================================

@app.get("/cameras")
def get_cameras():

    result = []

    for cam_id, runner in camera_runners.items():

        result.append({
            "cam_id": cam_id,
            "running": runner.running
        })

    return result

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "AI server running"
    }
