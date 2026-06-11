import cv2
import json
import os
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.config.settings import (
    ACTIVE_EVENT_NO_PLATE_MIN_FRAMES,
    ACTIVE_EVENT_PLATE_CONFIDENCE,
    ACTIVE_EVENT_SEND_INTERVAL_FRAMES,
    AI_PUBLIC_URL,
    DROP_LATE_FRAMES,
    EVENT_POST_QUEUE_LIMIT,
    MAX_FRAME_SKIP,
)


def resize_keep_ratio(frame, target_height=720):
    """
    Resize frame theo chiều cao target_height, giữ nguyên tỉ lệ.
    """
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    scale = target_height / float(h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


class CameraRunner:
    def __init__(self, cam_id, video_source, pipeline,
                 save_dir="./output", show=False,
                 drop_late_frames=DROP_LATE_FRAMES, max_frame_skip=MAX_FRAME_SKIP,
                 camera_config=None, camera_api_url=None,
                 send_vehicle_events=None):
        self.cam_id = cam_id
        self.video_source = video_source
        self.pipeline = pipeline
        self.save_dir = save_dir
        self.show = show
        self.drop_late_frames = drop_late_frames
        self.max_frame_skip = max_frame_skip
        self.camera_config = camera_config or {"id": cam_id}
        self.camera_api_url = camera_api_url
        self.event_image_base_url = (
            os.getenv("AI_PUBLIC_URL", AI_PUBLIC_URL) # 10.60.229.211
            .rstrip("/")
            + "/event-images"
        )
        if send_vehicle_events is None:
            self.send_vehicle_events = (
                os.getenv("SEND_VEHICLE_EVENTS", "true").lower()
                in ("1", "true", "yes", "on")
            )
        else:
            self.send_vehicle_events = bool(send_vehicle_events)

        self.cap = None
        self.writer = None

        self.frame_idx = 0
        self.running = False
        self.detect_started = False

        self.orig_fps = 30.0
        self.frame_interval = 1.0 / self.orig_fps

        self.latest_frame = None
        self.latest_frame_idx = -1

        self.on_finish = None
        self.sent_event_track_ids = set()
        self.queued_event_track_ids = set()
        self.event_executor = ThreadPoolExecutor(max_workers=1)
        self.event_futures = []
        self.event_post_queue_limit = int(
            os.getenv("EVENT_POST_QUEUE_LIMIT", str(EVENT_POST_QUEUE_LIMIT))
        )
        self.active_send_interval_frames = int(
            os.getenv(
                "ACTIVE_EVENT_SEND_INTERVAL_FRAMES",
                str(ACTIVE_EVENT_SEND_INTERVAL_FRAMES)
            )
        )
        self.good_plate_confidence = float(
            os.getenv(
                "ACTIVE_EVENT_PLATE_CONFIDENCE",
                str(ACTIVE_EVENT_PLATE_CONFIDENCE)
            )
        )
        self.no_plate_min_frames = int(
            os.getenv(
                "ACTIVE_EVENT_NO_PLATE_MIN_FRAMES",
                str(ACTIVE_EVENT_NO_PLATE_MIN_FRAMES)
            )
        )

        self.finalized = False

    def close_event_executor(self):
        for future in self.event_futures:
            try:
                future.result(timeout=5)
            except Exception as exc:
                print("[BE EVENT WAIT ERROR]", exc)
        self.event_futures = []
        self.event_executor.shutdown(wait=False)

    def get_result_age_frames(self, result):
        first_seen_frame = result.get("first_seen_frame")
        last_seen_frame = result.get("last_seen_frame")

        if first_seen_frame is None:
            return 0

        if last_seen_frame is None:
            last_seen_frame = self.frame_idx

        return max(0, int(last_seen_frame) - int(first_seen_frame) + 1)

    def should_send_result(self, result):
        track_id = result.get("track_id")
        if (
            track_id in self.sent_event_track_ids
            or track_id in self.queued_event_track_ids
        ):
            return False

        best_vehicle_img = result.get("best_vehicle_img")
        if best_vehicle_img is None or best_vehicle_img.size <= 0:
            return False

        plate_text = result.get("plate_text")
        plate_confidence = float(result.get("avg_confidence") or 0.0)
        successful_reads = int(result.get("successful_reads") or 0)

        has_good_plate = (
            bool(plate_text)
            and successful_reads >= 2
            and plate_confidence >= self.good_plate_confidence
        )
        if has_good_plate:
            return True

        age_frames = self.get_result_age_frames(result)
        read_attempts = int(result.get("read_attempts") or 0)
        detection_attempts = int(result.get("detection_attempts") or 0)
        status = result.get("status")
        is_finished = bool(result.get("finalized"))

        has_enough_plate_attempts = (
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
                    and has_enough_plate_attempts
                )
                or (
                    is_finished
                    and (
                        read_attempts >= 1
                        or detection_attempts >= 1
                        or status in ("no_plate", "unreadable", "low_confidence")
                    )
                )
            )
        )

        return has_low_confidence_plate or has_waited_for_plate

    def save_event_images(self, result):
        track_id = result.get("track_id")
        if track_id is None:
            return None, None

        image_dir = os.path.join(self.save_dir, "events", str(self.cam_id))
        os.makedirs(image_dir, exist_ok=True)

        image_path = None
        plate_image_path = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_prefix = f"track_{track_id}_{timestamp}"

        vehicle_img = result.get("best_vehicle_img")
        if vehicle_img is not None and vehicle_img.size > 0:
            image_filename = f"{file_prefix}_vehicle.jpg"
            image_file_path = os.path.join(image_dir, image_filename)
            cv2.imwrite(image_file_path, vehicle_img)
            image_path = (
                f"{self.event_image_base_url}/{self.cam_id}/{image_filename}"
            )

        plate_img = result.get("best_plate_img")
        if plate_img is not None and plate_img.size > 0:
            plate_filename = f"{file_prefix}_plate.jpg"
            plate_file_path = os.path.join(image_dir, plate_filename)
            cv2.imwrite(plate_file_path, plate_img)
            plate_image_path = (
                f"{self.event_image_base_url}/{self.cam_id}/{plate_filename}"
            )

        return image_path, plate_image_path

    def send_result_to_server(self, result):
        try:
            from src.app.event_client import (
                build_vehicle_event_payload,
                post_vehicle_event,
            )
        except Exception as exc:
            print("[BE EVENT IMPORT ERROR]", exc)
            return

        track_id = result.get("track_id")

        try:
            payload = build_vehicle_event_payload(
                self.camera_config,
                result
            )
            image_path, plate_image_path = self.save_event_images(result)
            payload["image_path"] = image_path
            payload["plate_image_path"] = plate_image_path

            response = post_vehicle_event(self.camera_api_url, payload)
            if response is not None:
                self.sent_event_track_ids.add(track_id)
                print(
                    "[BE EVENT SENT]",
                    f"camera={self.cam_id}",
                    f"track={track_id}",
                    f"plate={payload.get('plate')}",
                    f"status={payload.get('status')}",
                )
            else:
                print(
                    "[BE EVENT FAILED]",
                    f"camera={self.cam_id}",
                    f"track={track_id}",
                )
        except Exception as exc:
            print(
                "[BE EVENT ERROR]",
                f"camera={self.cam_id}",
                f"track={track_id}",
                exc,
            )
        finally:
            self.queued_event_track_ids.discard(track_id)

    def queue_result_to_server(self, result):
        self.event_futures = [
            future for future in self.event_futures
            if not future.done()
        ]

        if len(self.event_futures) >= self.event_post_queue_limit:
            print(
                "[BE EVENT SKIPPED]",
                f"camera={self.cam_id}",
                f"track={result.get('track_id')}",
                "reason=post_queue_full",
            )
            return

        track_id = result.get("track_id")
        self.queued_event_track_ids.add(track_id)
        future = self.event_executor.submit(
            self.send_result_to_server,
            result
        )
        self.event_futures.append(future)

    def send_results_to_server(self, results, eligible_only=True):
        if (
            not self.send_vehicle_events
            or not self.camera_api_url
            or not results
        ):
            return

        for result in results:
            track_id = result.get("track_id")
            if track_id is None:
                continue

            if eligible_only and not self.should_send_result(result):
                continue

            if (
                not eligible_only
                and (
                    track_id in self.sent_event_track_ids
                    or track_id in self.queued_event_track_ids
                )
            ):
                continue

            self.queue_result_to_server(result)

    def send_active_results_to_server(self):
        if (
            self.active_send_interval_frames <= 0
            or self.frame_idx % self.active_send_interval_frames != 0
        ):
            return

        results = self.pipeline.tracking_manager.export_all_results()
        self.send_results_to_server(results, eligible_only=True)

    # =====================================================
    # UNIQUE PATH
    # =====================================================

    @staticmethod
    def make_unique_path(path):
        if not os.path.exists(path):
            return path

        base, ext = os.path.splitext(path)
        i = 1
        while True:
            new_path = f"{base}_{i}{ext}"
            if not os.path.exists(new_path):
                return new_path
            i += 1

    # =====================================================
    # SETUP
    # =====================================================

    def setup(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open source: {self.video_source}")

        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_interval = 1.0 / max(self.orig_fps, 1.0)

        # Đọc 1 frame để biết kích thước sau resize
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise ValueError("Cannot read first frame from source")

        # frame_resized = resize_keep_ratio(frame, target_height=720)
        h, w = frame.shape[:2]

        os.makedirs(self.save_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # out_path = self.make_unique_path(
        #     os.path.join(
        #         self.save_dir,
        #         f"{self.cam_id}_tracked.mp4"
        #     )
        # )

        # self.writer = cv2.VideoWriter(
        #     out_path,
        #     fourcc,
        #     self.orig_fps,
        #     (w, h)
        # )

        # Đưa lại cap về frame đầu tiên
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.running = True
        self.detect_started = False
        self.frame_idx = 0

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run_loop(self):
        try:
            while self.running:
                start_time = time.perf_counter()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print(f"[INFO] Video ended: {self.cam_id}")
                    break

                # Resize giữ nguyên tỉ lệ về chiều cao 720
                # frame = resize_keep_ratio(frame, target_height=720)

                # Đồng bộ frame_index cho pipeline
                self.pipeline.frame_index = self.frame_idx

                out_frame, results = self.pipeline.process_frame_with_tracked_vehicles(frame)
                self.send_results_to_server(results, eligible_only=True)

                self.latest_frame = out_frame
                self.latest_frame_idx = self.frame_idx
                self.detect_started = True

                # Dọn track cũ theo chu kỳ (nếu cần)
                if (
                    self.frame_idx > 0
                    and self.active_send_interval_frames > 0
                    and self.frame_idx % self.active_send_interval_frames == 0
                ):
                    self.pipeline.tracking_manager.remove_expired_tracks(
                        self.frame_idx
                    )
                    # self.send_active_results_to_server()

                self.send_active_results_to_server()

                if self.writer is not None:
                    try:
                        self.writer.write(out_frame)
                    except Exception as e:
                        print("[WRITER ERROR]", e)

                if self.show:
                    cv2.imshow(f"Cam {self.cam_id}", out_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        self.stop()
                        break

                self.frame_idx += 1

                if self.drop_late_frames and self.cap is not None:
                    elapsed = time.perf_counter() - start_time
                    late_frames = int(elapsed / self.frame_interval) - 1
                    skip_count = max(0, min(self.max_frame_skip, late_frames))
                    for _ in range(skip_count):
                        if not self.cap.grab():
                            break
                        self.frame_idx += 1

        except Exception as e:
            print("[RUN LOOP ERROR]", e)

        finally:
            self.stop()
            self.finalize()
            if self.on_finish:
                self.on_finish(self.cam_id)

    # =====================================================
    # STOP
    # =====================================================

    def stop(self):
        self.running = False
        self.detect_started = False
        self.latest_frame = None
        self.latest_frame_idx = -1

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.show:
            try:
                cv2.destroyWindow(f"Cam {self.cam_id}")
            except:
                pass

    # =====================================================
    # FINALIZE
    # =====================================================

    def finalize(self):
        if self.finalized:
            return

        self.finalized = True

        self.pipeline.tracking_manager.finalize_all_active_tracks()

        results = self.pipeline.tracking_manager.export_all_results()
        self.send_results_to_server(results, eligible_only=False)
        self.close_event_executor()

        json_path = self.make_unique_path(
            os.path.join(
                self.save_dir,
                f"{self.cam_id}_tracked.json"
            )
        )

        print("[INFO] Saving JSON:", json_path)

        # with open(json_path, "w", encoding="utf-8") as f:
        #     json.dump(to_jsonable({
        #         "camera_id": self.cam_id,
        #         "video_source": self.video_source,
        #         "total_frames": self.frame_idx,
        #         "total_vehicles": len(results),
        #         "vehicles": results,
        #     }), f, ensure_ascii=False, indent=2)

        self.pipeline.tracking_manager.clear_finalized()

        return results
