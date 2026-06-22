import argparse
import json
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    OCR_MODEL_PATH,
    OUTPUT_DIR,
    PLATE_MODEL_PATH,
    VEHICLE_MODEL_PATH,
)
from src.main.PlateChar import PlateChar
from src.main.PlateDetector import PlateDetector
from src.main.VehicleDetector import VehicleDetector
from src.pipeline.PipelineManager import PipelineManager


class VideoOutputWriter:
    def __init__(self, output_path, fps, queue_size=120):
        self.output_path = Path(output_path)
        self.fps = max(float(fps or 0), 1.0)
        self.writer = None
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            name="video-output-writer",
            daemon=True,
        )
        self.thread.start()

    def __call__(self, camera_id, frame, results):
        if frame is None:
            return

        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                return
            try:
                self.queue.put_nowait(frame.copy())
            except queue.Full:
                return

    def _open_writer(self, frame):
        if self.writer is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")

    def _run(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._open_writer(frame)
                self.writer.write(frame)
            finally:
                self.queue.task_done()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the threaded vehicle and ANPR pipeline on one video."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="./dataset/vehicle/videos/27_075.mp4",
        help="Input video path.",
    )
    parser.add_argument(
        "--camera-id",
        default="video_test",
        help="Camera identifier used by the pipeline.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the rendered video. Press Esc to stop.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rendered frames to an MP4 file.",
    )
    parser.add_argument(
        "--save-event-images",
        action="store_true",
        help="Save the best vehicle and plate images prepared for server events.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for rendered video and JSON results.",
    )
    return parser.parse_args()


def validate_file(path, label):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def get_video_fps(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        return capture.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        capture.release()


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
            if not isinstance(item, np.ndarray)
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_event_images(results, output_dir, camera_id, run_id):
    image_dir = Path(output_dir) / "events" / camera_id / run_id
    image_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for result in results:
        track_id = result.get("track_id")
        if track_id is None:
            continue

        paths = {
            "track_id": track_id,
            "vehicle_image": None,
            "plate_image": None,
        }

        vehicle_img = result.get("best_vehicle_img")
        if vehicle_img is not None and vehicle_img.size > 0:
            vehicle_path = image_dir / f"track_{track_id}_vehicle.jpg"
            if cv2.imwrite(str(vehicle_path), vehicle_img):
                paths["vehicle_image"] = str(vehicle_path.resolve())

        plate_img = result.get("best_plate_img")
        if plate_img is not None and plate_img.size > 0:
            plate_path = image_dir / f"track_{track_id}_plate.jpg"
            if cv2.imwrite(str(plate_path), plate_img):
                paths["plate_image"] = str(plate_path.resolve())

        if paths["vehicle_image"] or paths["plate_image"]:
            saved.append(paths)

    return image_dir, saved


def main():
    args = parse_args()
    video_path = validate_file(args.video, "Video")
    vehicle_path = validate_file(VEHICLE_MODEL_PATH, "Vehicle model")
    plate_path = validate_file(PLATE_MODEL_PATH, "Plate model")
    ocr_path = validate_file(OCR_MODEL_PATH, "OCR model")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_stem = (
        f"{args.camera_id}_{video_path.stem}_threaded_{run_id}"
    )

    writer = None
    if args.save_video:
        writer = VideoOutputWriter(
            output_dir / f"{output_stem}.mp4",
            get_video_fps(video_path),
        )

    print(f"[INFO] Video: {video_path}")
    print("[INFO] Loading models...")
    manager = PipelineManager(
        vehicle_model=VehicleDetector(str(vehicle_path)),
        plate_model=PlateDetector(str(plate_path)),
        char_model=PlateChar(str(ocr_path)),
        show=args.show,
        on_results=writer,
    )
    manager.add_camera(args.camera_id, str(video_path))

    try:
        all_results = manager.run_until_complete()
    finally:
        if writer is not None:
            writer.close()

    errors = []
    while True:
        error = manager.get_error(timeout=0)
        if error is None:
            break
        worker_name, camera_id, exception = error
        errors.append(
            {
                "worker": worker_name,
                "camera_id": camera_id,
                "error": repr(exception),
            }
        )

    camera_results = all_results.get(args.camera_id, [])
    event_image_dir = None
    saved_event_images = []
    if args.save_event_images:
        event_image_dir, saved_event_images = save_event_images(
            camera_results,
            output_dir,
            args.camera_id,
            run_id,
        )

    json_path = output_dir / f"{output_stem}.json"
    payload = {
        "run_id": run_id,
        "camera_id": args.camera_id,
        "video": str(video_path),
        "vehicle_count": len(camera_results),
        "vehicles": [
            to_jsonable(result)
            for result in camera_results
        ],
        "errors": errors,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] Vehicles: {len(camera_results)}")
    # for result in camera_results:
    #     print(
    #         "[RESULT]",
    #         f"track={result.get('track_id')}",
    #         f"type={result.get('vehicle_type')}",
    #         f"plate={result.get('plate_text') or '-'}",
    #         f"status={result.get('status')}",
    #     )
    # print(f"[INFO] JSON: {json_path.resolve()}")
    if writer is not None:
        print(f"[INFO] Video output: {writer.output_path.resolve()}")
    if event_image_dir is not None:
        print(
            f"[INFO] Event images: {event_image_dir.resolve()} "
            f"({len(saved_event_images)} tracks)"
        )
    if errors:
        print(f"[WARNING] Worker errors: {len(errors)}")


if __name__ == "__main__":
    main()
