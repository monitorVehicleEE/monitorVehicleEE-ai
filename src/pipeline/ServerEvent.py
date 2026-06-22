import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2

from src.app.event_client import (
    build_vehicle_event_payload,
    post_vehicle_event,
)
from src.config.settings import (
    ACTIVE_EVENT_NO_PLATE_MIN_FRAMES,
    ACTIVE_EVENT_PLATE_CONFIDENCE,
    ACTIVE_EVENT_SEND_INTERVAL_FRAMES,
    AI_PUBLIC_URL,
    CAMERA_API_URL,
    EVENT_IMAGES_DIR,
    EVENT_POST_QUEUE_LIMIT,
)


class ServerEvent:

    def __init__(
        self,
        camera_api_url=None,
        event_images_dir=None,
        ai_public_url=None,
        queue_limit=None,
        send_interval=None,
        good_plate_confidence=None,
        no_plate_min_frames=None,
        payload_builder=build_vehicle_event_payload,
        poster=post_vehicle_event,
    ):
        self.camera_api_url = (
            camera_api_url
            or os.getenv("CAMERA_API_URL", CAMERA_API_URL)
        )
        self.event_images_dir = Path(
            event_images_dir
            or os.getenv("EVENT_IMAGES_DIR", EVENT_IMAGES_DIR)
        )
        self.event_image_base_url = (
            ai_public_url
            or os.getenv("AI_PUBLIC_URL", AI_PUBLIC_URL)
        ).rstrip("/") + "/event-images"
        self.queue_limit = int(
            queue_limit
            if queue_limit is not None
            else os.getenv(
                "EVENT_POST_QUEUE_LIMIT",
                str(EVENT_POST_QUEUE_LIMIT),
            )
        )
        self.send_interval = int(
            send_interval
            if send_interval is not None
            else os.getenv(
                "ACTIVE_EVENT_SEND_INTERVAL_FRAMES",
                str(ACTIVE_EVENT_SEND_INTERVAL_FRAMES),
            )
        )
        self.good_plate_confidence = float(
            good_plate_confidence
            if good_plate_confidence is not None
            else os.getenv(
                "ACTIVE_EVENT_PLATE_CONFIDENCE",
                str(ACTIVE_EVENT_PLATE_CONFIDENCE),
            )
        )
        self.no_plate_min_frames = int(
            no_plate_min_frames
            if no_plate_min_frames is not None
            else os.getenv(
                "ACTIVE_EVENT_NO_PLATE_MIN_FRAMES",
                str(ACTIVE_EVENT_NO_PLATE_MIN_FRAMES),
            )
        )
        self.payload_builder = payload_builder
        self.poster = poster

        self.cameras = {}
        self.sent_track_ids = {}
        self.queued_track_ids = {}
        self.callback_counts = {}
        
        # Tăng số lượng worker lên 4 để xử lý song song, tránh tắc nghẽn hàng đợi
        self.executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="be-event",
        )
        self.futures = []
        self.lock = threading.RLock()
        self.closed = False

    def register_camera(self, camera_id, camera_config=None, enabled=True):
        config = dict(camera_config or {})
        config["id"] = config.get("id", camera_id)
        key = str(camera_id)
        with self.lock:
            self.cameras[key] = {
                "config": config,
                "enabled": bool(enabled),
            }
            self.sent_track_ids.setdefault(key, set())
            self.queued_track_ids.setdefault(key, set())
            self.callback_counts.setdefault(key, 0)

    def set_enabled(self, camera_id, enabled):
        key = str(camera_id)
        with self.lock:
            camera = self.cameras.get(key)
            if camera is not None:
                camera["enabled"] = bool(enabled)

    @staticmethod
    def _result_age_frames(result):
        first_seen = result.get("first_seen_frame")
        last_seen = result.get("last_seen_frame")
        if first_seen is None:
            return 0
        if last_seen is None:
            last_seen = first_seen
        return max(0, int(last_seen) - int(first_seen) + 1)

    def _should_send(self, camera_id, result):
        track_id = result.get("track_id")
        if track_id is None:
            return False

        key = str(camera_id)
        with self.lock:
            if (
                track_id in self.sent_track_ids.get(key, set())
                or track_id in self.queued_track_ids.get(key, set())
            ):
                return False

        vehicle_img = result.get("best_vehicle_img")
        if vehicle_img is None or vehicle_img.size <= 0:
            return False

        plate_text = result.get("plate_text")
        plate_confidence = float(result.get("avg_confidence") or 0.0)
        successful_reads = int(result.get("successful_reads") or 0)
        if (
            plate_text
            and successful_reads >= 2
            and plate_confidence >= self.good_plate_confidence
        ):
            return True

        age_frames = self._result_age_frames(result)
        read_attempts = int(result.get("read_attempts") or 0)
        detection_attempts = int(result.get("detection_attempts") or 0)
        status = result.get("status")
        is_finished = bool(result.get("finalized"))
        has_enough_attempts = (
            read_attempts >= 2
            or detection_attempts >= 2
            or status in ("no_plate", "unreadable", "low_confidence")
        )
        has_low_confidence_plate = (
            bool(plate_text)
            and successful_reads >= 1
            and age_frames >= self.no_plate_min_frames
        )
        has_waited_for_plate = (
            not plate_text
            and (
                (
                    age_frames >= self.no_plate_min_frames
                    and has_enough_attempts
                )
                or (
                    is_finished
                    and (
                        read_attempts >= 1
                        or detection_attempts >= 1
                        or status
                        in ("no_plate", "unreadable", "low_confidence")
                    )
                )
            )
        )
        return has_low_confidence_plate or has_waited_for_plate

    def _save_images(self, camera_id, result):
        track_id = result.get("track_id")
        if track_id is None:
            return None, None

        camera_dir = self.event_images_dir / str(camera_id)
        camera_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = f"track_{track_id}_{timestamp}"

        image_path = None
        plate_image_path = None
        vehicle_img = result.get("best_vehicle_img")
        if vehicle_img is not None and vehicle_img.size > 0:
            filename = f"{prefix}_vehicle.jpg"
            if cv2.imwrite(str(camera_dir / filename), vehicle_img):
                image_path = (
                    f"{self.event_image_base_url}/{camera_id}/{filename}"
                )

        plate_img = result.get("best_plate_img")
        if plate_img is not None and plate_img.size > 0:
            filename = f"{prefix}_plate.jpg"
            if cv2.imwrite(str(camera_dir / filename), plate_img):
                plate_image_path = (
                    f"{self.event_image_base_url}/{camera_id}/{filename}"
                )

        return image_path, plate_image_path

    def _send(self, camera_id, result):
        key = str(camera_id)
        track_id = result.get("track_id")
        try:
            with self.lock:
                camera = self.cameras.get(key)
                if camera is None:
                    return
                camera_config = dict(camera["config"])

            payload = self.payload_builder(camera_config, result)
            image_path, plate_image_path = self._save_images(
                camera_id,
                result,
            )
            payload["image_path"] = image_path
            payload["plate_image_path"] = plate_image_path
            response = self.poster(self.camera_api_url, payload)
            if response is not None:
                with self.lock:
                    self.sent_track_ids[key].add(track_id)
            else:
                print(
                    "[BE EVENT FAILED]",
                    f"camera={camera_id}",
                    f"track={track_id}",
                )
        except Exception as exc:
            print(
                "[BE EVENT ERROR]",
                f"camera={camera_id}",
                f"track={track_id}",
                exc,
            )
        finally:
            with self.lock:
                self.queued_track_ids.get(key, set()).discard(track_id)

    def _queue_result(self, camera_id, result):
        key = str(camera_id)
        track_id = result.get("track_id")
        with self.lock:
            self.futures = [
                future for future in self.futures
                if not future.done()
            ]
            if len(self.futures) >= self.queue_limit:
                print(
                    "[BE EVENT SKIPPED]",
                    f"camera={camera_id}",
                    f"track={track_id}",
                    "reason=post_queue_full",
                )
                return
            
            self.queued_track_ids[key].add(track_id)
            
            #Deep Copy ma trận ảnh trên luồng chính để luồng phụ ghi file an toàn ---
            safe_result = dict(result)
            if "best_vehicle_img" in result and result["best_vehicle_img"] is not None:
                safe_result["best_vehicle_img"] = result["best_vehicle_img"].copy()
            if "best_plate_img" in result and result["best_plate_img"] is not None:
                safe_result["best_plate_img"] = result["best_plate_img"].copy()
            # --------------------------------------------------------------------------------------

            future = self.executor.submit(
                self._send,
                camera_id,
                safe_result,
            )
            self.futures.append(future)

    def send_results(self, camera_id, results, eligible_only=True):
        key = str(camera_id)
        with self.lock:
            camera = self.cameras.get(key)
            if (
                self.closed
                or camera is None
                or not camera["enabled"]
                or not self.camera_api_url
            ):
                return

        for result in results or []:
            track_id = result.get("track_id")
            if track_id is None:
                continue
            if eligible_only:
                if not self._should_send(camera_id, result):
                    continue
            else:
                with self.lock:
                    if (
                        track_id in self.sent_track_ids[key]
                        or track_id in self.queued_track_ids[key]
                    ):
                        continue
            self._queue_result(camera_id, result)

    def __call__(self, camera_id, frame, results):
        key = str(camera_id)
        with self.lock:
            if key not in self.cameras:
                return
            self.callback_counts[key] += 1
            callback_count = self.callback_counts[key]

        if (
            self.send_interval > 0
            and callback_count % self.send_interval == 0
        ):
            self.send_results(camera_id, results, eligible_only=True)

    def close(self, final_results=None):
        with self.lock:
            if self.closed:
                return

        for camera_id, results in (final_results or {}).items():
            self.send_results(camera_id, results, eligible_only=False)

        with self.lock:
            futures = list(self.futures)
        for future in futures:
            try:
                future.result(timeout=5)
            except Exception as exc:
                print("[BE EVENT WAIT ERROR]", exc)

        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=False)