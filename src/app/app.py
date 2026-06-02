from threading import Thread, Lock
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles

from src.main.MainPipeline import MainPipeline
from src.main.CameraRunner import CameraRunner
from src.main.VehicleTracker import VehicleTracker
from src.main.PlateDetector import PlateDetector
from src.main.PlateChar import PlateChar
from src.main.VehicleDetector import VehicleDetector


import cv2
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

STREAM_TIMING_LOG = False
STREAM_JPEG_QUALITY = 75
START_READY_TIMEOUT = 60.0
CAMERA_API_URL = os.getenv("CAMERA_API_URL", "http://localhost:8000")
VIDEO_BASE_DIR = os.getenv("VIDEO_BASE_DIR", "./dataset/vehicle/videos")

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()
os.makedirs("./dataset/output_test/events", exist_ok=True)
app.mount(
    "/event-images",
    StaticFiles(directory="./dataset/output_test/events"),
    name="event-images"
)

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

detector_vehicle = VehicleDetector("./model/pytorch/vehicle/best.pt", device=0)

detector_plate = PlateDetector("./model/pytorch/plate/best.pt", device=0)

detector_char = PlateChar("./model/pytorch/char/best.pt", device=0)

# =========================================================
# CAMERA STORE
# =========================================================

camera_runners = {}

def wait_until_detect_started(runner, timeout=START_READY_TIMEOUT):

    deadline = time.perf_counter() + timeout

    while runner.running and not runner.detect_started:

        if time.perf_counter() >= deadline:
            return False

        time.sleep(0.05)

    return runner.running and runner.detect_started

def unwrap_camera_payload(payload):
    if not isinstance(payload, dict):
        return None

    for key in ("data", "camera", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested

    return payload


def infer_source_type(source_path):
    source = str(source_path or "").strip().lower()

    if source.startswith("rtsp://"):
        return "rtsp"

    if source.startswith("http://"):
        return "http"

    if source.startswith("https://"):
        return "https"

    return "file"


def normalize_camera_config(cam_id, payload):
    camera = unwrap_camera_payload(payload)
    if not camera:
        return None

    source_path = (
        camera.get("source_path")
        or camera.get("sourcePath")
        or camera.get("source")
        or camera.get("url")
        or camera.get("video_url")
        or camera.get("videoUrl")
        or camera.get("stream_url")
        or camera.get("streamUrl")
        or camera.get("rtsp_url")
        or camera.get("rtspUrl")
        or camera.get("path")
    )
    source_type = (
        camera.get("source_type")
        or camera.get("sourceType")
        or camera.get("type")
        or infer_source_type(source_path)
    )

    normalized = dict(camera)
    normalized["id"] = normalized.get("id") or cam_id
    normalized["source_type"] = str(source_type or "").lower()
    normalized["source_path"] = str(source_path or "").strip()

    if "camera_role" not in normalized and "cameraRole" in normalized:
        normalized["camera_role"] = normalized["cameraRole"]

    return normalized


def fetch_camera_config(cam_id):
    url = f"{CAMERA_API_URL.rstrip('/')}/cameras/{cam_id}"

    try:
        with urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            camera = normalize_camera_config(cam_id, payload)
            if camera:
                return camera

            raise HTTPException(
                status_code=502,
                detail="backend camera api returned invalid camera payload"
            )
    except HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(
                status_code=404,
                detail=f"camera not found in backend: {cam_id}"
            )

        raise HTTPException(
            status_code=502,
            detail=f"backend camera api error: {exc.code}"
        )
    except URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"cannot connect to backend camera api: {exc.reason}"
        )


def resolve_camera_source(camera):
    source_type = str(camera.get("source_type") or "").lower()
    source_path = str(camera.get("source_path") or "").strip()

    if not source_path:
        raise HTTPException(
            status_code=400,
            detail="camera source_path is empty"
        )

    if source_type in {"rtsp", "http", "https"}:
        return source_path

    if source_type != "file":
        raise HTTPException(
            status_code=400,
            detail=f"unsupported camera source_type: {source_type}"
        )

    if os.path.isabs(source_path):
        candidate_paths = [source_path]
    else:
        candidate_paths = [
            source_path,
            os.path.join(VIDEO_BASE_DIR, source_path),
        ]

    for candidate_path in candidate_paths:
        normalized_path = os.path.normpath(candidate_path)
        if os.path.isfile(normalized_path):
            return normalized_path

    raise HTTPException(
        status_code=404,
        detail=f"video file not found: {source_path}"
    )

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
@app.post("/start-stream/{cam_id}/{send_event}")
def start_stream(
    cam_id: str,
    send_event: bool = True,
    camera_payload: dict = Body(default=None)
):

    with camera_lock:

        # camera đang chạy
        if cam_id in camera_runners:

            runner = camera_runners[cam_id]

            if runner.running:
                if send_event:
                    runner.send_vehicle_events = True

                ready = wait_until_detect_started(runner)

                return {
                    "status": "already_running" if ready else "starting",
                    "cam_id": cam_id,
                    "ready": ready,
                    "send_vehicle_events": runner.send_vehicle_events
                }

        camera = normalize_camera_config(cam_id, camera_payload)
        if camera is None:
            camera = fetch_camera_config(cam_id)

        video_source = resolve_camera_source(camera)

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
            max_frame_skip=1,
            camera_config=camera,
            camera_api_url=CAMERA_API_URL,
            send_vehicle_events=send_event)

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

        ready = wait_until_detect_started(runner)

        return {
            "status": "started" if ready else "starting",
            "cam_id": cam_id,
            "source": video_source,
            "ready": ready,
            "send_vehicle_events": runner.send_vehicle_events
        }

# =========================================================
# STOP STREAM
# =========================================================

@app.post("/stop-stream/{cam_id}")
@app.post("/stop-stream/{cam_id}/{send_event}")
def stop_stream(cam_id: str, send_event: bool = True):

    with camera_lock:

        runner = camera_runners.get(cam_id)

        if runner is None:

            return {
                "error": "camera not found"
            }

        if not send_event and runner.send_vehicle_events:
            return {
                "status": "event_stream_running",
                "cam_id": cam_id,
                "stopped": False,
                "send_vehicle_events": runner.send_vehicle_events
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
            "running": False,
            "send_vehicle_events": False
        }

    return {
        "running": runner.running and runner.detect_started,
        "send_vehicle_events": runner.send_vehicle_events
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
            "running": runner.running and runner.detect_started
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
