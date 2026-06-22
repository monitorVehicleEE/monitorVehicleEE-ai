import json
import os
import queue
import time
from threading import Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import cv2
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config.settings import (
    CAMERA_API_URL as DEFAULT_CAMERA_API_URL,
    DEVICE,
    EVENT_IMAGES_DIR,
    JPEG_QUALITY,
    OCR_MODEL_PATH,
    PLATE_MODEL_PATH,
    SHOW_WINDOW,
    START_READY_TIMEOUT,
    STREAM_TIMING_LOG,
    STREAM_WIDTH,
    VEHICLE_MODEL_PATH,
    VIDEO_BASE_DIR as DEFAULT_VIDEO_BASE_DIR,
)
from src.main.PlateChar import PlateChar
from src.main.PlateDetector import PlateDetector
from src.main.VehicleDetector import VehicleDetector
from src.pipeline.PipelineManager import PipelineManager
from src.pipeline import ServerEvent

STREAM_JPEG_QUALITY = JPEG_QUALITY
STREAM_CACHE_LIMIT = 4
STREAM_DEFAULT_FPS_CAP = float(os.getenv("STREAM_FPS_CAP", "15"))
CAMERA_API_URL = os.getenv("CAMERA_API_URL", DEFAULT_CAMERA_API_URL)
VIDEO_BASE_DIR = os.getenv("VIDEO_BASE_DIR", DEFAULT_VIDEO_BASE_DIR)

app = FastAPI()
os.makedirs(EVENT_IMAGES_DIR, exist_ok=True)
app.mount(
    "/event-images",
    StaticFiles(directory=EVENT_IMAGES_DIR),
    name="event-images",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_lock = Lock()
camera_store = {}

# Khởi tạo duy nhất 1 lần để dùng chung tài nguyên GPU/RAM giữa các luồng
detector_vehicle = VehicleDetector(VEHICLE_MODEL_PATH, device=DEVICE)
detector_plate = PlateDetector(PLATE_MODEL_PATH, device=DEVICE)
detector_char = PlateChar(OCR_MODEL_PATH, device=DEVICE)
server_event = ServerEvent(camera_api_url=CAMERA_API_URL)


class FastStreamBroadcaster:
    """
    Callback chuẩn từ run_video.py giúp hứng trọn vẹn frame và kết quả từ RenderWorker
    """
    def __init__(self, queue_size=120):
        self.queue = queue.Queue(maxsize=queue_size)
        self.is_stopped = False

    def __call__(self, camera_id, frame, results):
        if self.is_stopped:
            return
        if frame is None:
            # Đẩy tín hiệu kết thúc video vào queue
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self.queue.get_nowait()  # Đẩy frame cũ ra nếu hàng đợi đầy
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def stop(self):
        self.is_stopped = True
        # Đẩy tín hiệu None để giải phóng lệnh chặn .get() lập tức
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass


def unwrap_camera_payload(payload):
    if not isinstance(payload, dict): return None
    for key in ("data", "camera", "result"):
        nested = payload.get(key)
        if isinstance(nested, dict): return nested
    return payload


def infer_source_type(source_path):
    source = str(source_path or "").strip().lower()
    if source.startswith("rtsp://"): return "rtsp"
    if source.startswith("http://"): return "http"
    if source.startswith("https://"): return "https"
    return "file"


def normalize_camera_config(cam_id, payload):
    camera = unwrap_camera_payload(payload)
    if not camera: return None

    source_path = (
        camera.get("source_path") or camera.get("sourcePath") or camera.get("source") or
        camera.get("url") or camera.get("video_url") or camera.get("videoUrl") or
        camera.get("stream_url") or camera.get("streamUrl") or camera.get("rtsp_url") or
        camera.get("rtspUrl") or camera.get("path")
    )
    source_type = (
        camera.get("source_type") or camera.get("sourceType") or camera.get("type") or infer_source_type(source_path)
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
            if camera: return camera
            raise HTTPException(status_code=502, detail="invalid camera payload")
    except HTTPError as exc:
        if exc.code == 404:
            raise HTTPException(status_code=404, detail=f"camera not found: {cam_id}")
        raise HTTPException(status_code=502, detail=f"backend api error: {exc.code}")
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"cannot connect to backend: {exc.reason}")


def resolve_camera_source(camera):
    source_type = str(camera.get("source_type") or "").lower()
    source_path = str(camera.get("source_path") or "").strip()
    if not source_path:
        raise HTTPException(status_code=400, detail="camera source_path is empty")

    if source_type in {"rtsp", "http", "https"}:
        return source_path
    if source_type != "file":
        raise HTTPException(status_code=400, detail=f"unsupported source_type: {source_type}")

    candidate_paths = [source_path] if os.path.isabs(source_path) else [source_path, os.path.join(VIDEO_BASE_DIR, source_path)]
    for candidate_path in candidate_paths:
        normalized_path = os.path.normpath(candidate_path)
        if os.path.isfile(normalized_path):
            return normalized_path

    raise HTTPException(status_code=404, detail=f"video file not found: {source_path}")


def resize_keep_ratio(frame, target_width):
    height, width = frame.shape[:2]
    if width <= 0: return frame
    scale = target_width / width
    return cv2.resize(frame, (target_width, int(height * scale)))


def get_source_fps(source):
    if not isinstance(source, str): return 15.0
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened(): return 15.0
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        return min(float(fps), 30.0) if fps > 0 else 15.0
    finally:
        capture.release()


def get_stream_fps(source):
    source_fps = get_source_fps(source)
    if STREAM_DEFAULT_FPS_CAP <= 0: return source_fps
    return min(source_fps, STREAM_DEFAULT_FPS_CAP)


def generate_frames_from_broadcaster(cam_id, broadcaster: FastStreamBroadcaster, width=STREAM_WIDTH):
    width_key = int(width or STREAM_WIDTH)
    while True:
        # Nếu broadcaster báo dừng, kết thúc generator ngay lập tức
        if broadcaster.is_stopped:
            break
            
        try:
            frame = broadcaster.queue.get(timeout=0.5)
        except queue.Empty:
            # Kiểm tra xem camera còn nằm trong store không, nếu đã bị xóa thì thoát vòng lặp
            with camera_lock:
                if cam_id not in camera_store:
                    break
            continue

        # Nhận tín hiệu kết thúc video (Sentinel value)
        if frame is None:
            print(f"[INFO] Video stream ended for camera {cam_id}")
            break

        try:
            height, current_width = frame.shape[:2]
            if width is not None and current_width != width_key:
                frame = resize_keep_ratio(frame, width_key)

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY],
            )
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        except Exception as exc:
            print("[STREAM ERROR]", exc)
            break

    # Dọn dẹp camera khỏi store sau khi kết thúc luồng đọc video file hoàn tất
    with camera_lock:
        camera_store.pop(cam_id, None)


# ==================== ENDPOINTS CỐ ĐỊNH TÊN NHẬN ====================

@app.get("/stream/{cam_id}")
def stream(cam_id: str, request: Request):
    try:
        width = int(request.query_params.get("w", STREAM_WIDTH))
    except Exception:
        width = STREAM_WIDTH

    with camera_lock:
        camera = camera_store.get(cam_id)
        if camera is None:
            return {"error": "camera not found"}
        broadcaster = camera["broadcaster"]

    return StreamingResponse(
        generate_frames_from_broadcaster(cam_id, broadcaster, width),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/start-stream/{cam_id}")
@app.post("/start-stream/{cam_id}/{send_event}")
def start_stream(
    cam_id: str,
    send_event: bool = True,
    camera_payload: dict = Body(default=None),
):
    with camera_lock:
        if cam_id in camera_store:
            if send_event:
                camera_store[cam_id]["send_vehicle_events"] = True
                server_event.set_enabled(cam_id, True)
            return {
                "status": "already_running",
                "cam_id": cam_id,
                "ready": True,
                "send_vehicle_events": camera_store[cam_id]["send_vehicle_events"],
            }

    camera = normalize_camera_config(cam_id, camera_payload)
    if camera is None:
        camera = fetch_camera_config(cam_id)
    video_source = resolve_camera_source(camera)
    source_fps = get_source_fps(video_source)
    stream_fps = get_stream_fps(video_source)

    broadcaster = FastStreamBroadcaster()

    manager = PipelineManager(
        vehicle_model=detector_vehicle,
        plate_model=detector_plate,
        char_model=detector_char,
        show=SHOW_WINDOW,
        on_results=broadcaster, 
        event_publisher=server_event,
    )
    manager.add_camera(cam_id, video_source, camera_config=camera, send_vehicle_events=send_event)
    
    def run_pipeline_task(pipeline_mgr, cid, b_caster):
        try:
            pipeline_mgr.run_until_complete()
        except Exception as e:
            print(f"[PIPELINE ERROR] Camera {cid}: {e}")
        finally:
            # Khi chạy xong video, kích hoạt dừng broadcaster để giải phóng các API endpoint và generator
            b_caster.stop()

    t = Thread(target=run_pipeline_task, args=(manager, cam_id, broadcaster), name=f"pipeline-{cam_id}", daemon=True)
    t.start()

    with camera_lock:
        camera_store[cam_id] = {
            "manager": manager,  
            "camera": camera,
            "source": video_source,
            "send_vehicle_events": bool(send_event),
            "frame_interval": 1.0 / stream_fps,
            "stream_fps": stream_fps,
            "source_fps": source_fps,
            "broadcaster": broadcaster,
            "thread": t,
            "stream_stats": {"frames_served": 0, "cache_hits": 0, "cache_misses": 0}
        }

    print(f"[INFO] Started isolated offline-speed pipeline for camera: {cam_id}")
    return {
        "status": "started",
        "cam_id": cam_id,
        "source": video_source,
        "ready": True,
        "send_vehicle_events": bool(send_event),
        "stream_fps": stream_fps,
        "source_fps": source_fps,
    }


@app.post("/stop-stream/{cam_id}")
@app.post("/stop-stream/{cam_id}/{send_event}")
def stop_stream(cam_id: str, send_event: bool = True):
    with camera_lock:
        camera = camera_store.get(cam_id)
        if camera is None:
            return {"status": "already_stopped", "cam_id": cam_id}
        if not send_event and camera["send_vehicle_events"]:
            return {
                "status": "event_stream_running",
                "cam_id": cam_id,
                "stopped": False,
                "send_vehicle_events": camera["send_vehicle_events"],
            }

    with camera_lock:
        camera_data = camera_store.pop(cam_id, None)

    if camera_data is not None:
        camera_data["broadcaster"].stop()  # Ngắt block queue ngay lập tức
        manager = camera_data["manager"]
        manager.stop_camera(cam_id)
        manager.stop()

    print(f"[INFO] Stopped pipeline camera: {cam_id}")
    return {"status": "stopped", "cam_id": cam_id}


@app.get("/camera-status/{cam_id}")
def camera_status(cam_id: str):
    with camera_lock:
        camera = camera_store.get(cam_id)
        if camera is None:
            return {"running": False, "send_vehicle_events": False}
        
        manager = camera["manager"]
        is_running = manager.get_latest_frame(cam_id) is not None
        return {
            "running": is_running,
            "send_vehicle_events": camera["send_vehicle_events"],
        }


@app.get("/cameras")
def get_cameras():
    with camera_lock:
        items = list(camera_store.items())
    
    results = []
    for cam_id, camera in items:
        manager = camera["manager"]
        is_running = manager.get_latest_frame(cam_id) is not None
        results.append({"cam_id": cam_id, "running": is_running})
    return results


@app.get("/pipeline-stats/{cam_id}")
def pipeline_stats(cam_id: str):
    with camera_lock:
        camera = camera_store.get(cam_id)
    if camera is None:
        return {"camera_id": cam_id, "pipeline": {}, "stream": {}}
        
    manager = camera["manager"]
    return {
        "camera_id": cam_id,
        "pipeline": manager.get_stats(cam_id),
        "stream": dict(camera["stream_stats"]),
    }


@app.get("/")
def root():
    return {"message": "AI server running with high-performance broadcaster layout"}


@app.on_event("shutdown")
def shutdown_pipeline():
    with camera_lock:
        items = list(camera_store.values())
    for camera in items:
        camera["broadcaster"].stop()
        manager = camera["manager"]
        manager.stop()
        manager.join()