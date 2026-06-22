import threading
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from src.config.settings import (
    MAX_CACHED_FRAMES,
    MAX_OCR_READS,
    MAX_PLATE_FRAME_CHECKS,
    MAX_PLATE_SAMPLE_RETRIES,
    OCR_STEP,
    PLATE_STEP,
    TRACK_EXPIRE_FRAMES,
    TRACK_LOST_APPEND_FRAMES,
    TRACK_MAX_HISTORY,
    TRACK_MIN_LENGTH,
    TRACK_MIN_VOTES,
    VEHICLE_SAMPLE_STEP,
    VEHICLE_STEP,
)
from src.main.TrackingManager import TrackingManager
from src.pipeline.FrameData import FrameData
from src.pipeline.ResultTypes import OCRResult, PlateResult, VehicleResult
from src.pipeline.SharedsState import put_if_available, put_latest
from src.pipeline.TasksType import CharTask, PlateTask, VehicleTask
from src.main.VehicleTracker import VehicleTracker


@dataclass
class CameraRuntime:
    tracker: object
    tracking_manager: TrackingManager
    latest_vehicles: list = field(default_factory=list)
    last_vehicle_result_frame: int = -1
    pending_plate: set = field(default_factory=set)
    pending_ocr: set = field(default_factory=set)
    frame_cache: dict = field(default_factory=dict)


class TrackingWorker(threading.Thread):
    def __init__(
        self,
        state,
        tracker_factory: Optional[Callable] = None,
        tracking_manager_factory: Optional[Callable] = None,
        # Thường dùng cho: UI, stream frame ra API/websocket, lưu ảnh/video output, đẩy event sang server khác debug/log realtime
        # để pipeline gọi ngược ra ngoài khi có kết quả mới.
        # Nó giúp tách phần xử lý AI khỏi phần UI / streaming / event gửi server.
        on_results=None,
    ):
        super().__init__(name="tracking-results", daemon=True)
        self.state = state
        self.tracker_factory = tracker_factory or self._default_tracker_factory
        self.tracking_manager_factory = (
            tracking_manager_factory or self._default_tracking_manager
        )
        self.on_results = on_results
        self.cameras: Dict[str, CameraRuntime] = {} # self.cameras = {"cam_1": CameraRuntime(...),}
        # mỗi CameraRuntime giữ dữ liệu độc lập trong class CameraRuntime
        self.camera_order = []
        self.camera_cursor = 0
        self.camera_lock = threading.RLock()

    @staticmethod
    def _default_tracking_manager():
        return TrackingManager(
            min_length=TRACK_MIN_LENGTH,
            min_votes=TRACK_MIN_VOTES,
            max_history=TRACK_MAX_HISTORY,
            expire_frames=TRACK_EXPIRE_FRAMES,
        )

    @staticmethod
    def _default_tracker_factory():
        return VehicleTracker()

    # def register_camera(self, camera_id):
    #     self.cameras[camera_id] = CameraRuntime(
    #         tracker=self.tracker_factory(), # theo dõi vị trí và cấp track_id
    #         tracking_manager=self.tracking_manager_factory(), # lưu lịch sử track, kết quả OCR, voting và trạng thái biển số.
    #     )
    #     self.camera_order.append(camera_id)

    # lấy tracking_manager
    def register_camera(self, camera_id):
        with self.camera_lock:
            if camera_id in self.cameras:
                raise ValueError(f"Camera already registered: {camera_id}")
            self.cameras[camera_id] = CameraRuntime(
                tracker=self.tracker_factory(),
                tracking_manager=self.tracking_manager_factory(),
            )
            self.camera_order.append(camera_id)

    def unregister_camera(self, camera_id):
        with self.camera_lock:
            runtime = self.cameras.pop(camera_id, None)
            if camera_id in self.camera_order:
                self.camera_order.remove(camera_id)
            if self.camera_order:
                self.camera_cursor %= len(self.camera_order)
            else:
                self.camera_cursor = 0

        if runtime is not None:
            runtime.tracking_manager.finalize_all_active_tracks()
            runtime.pending_plate.clear()
            runtime.pending_ocr.clear()
            runtime.frame_cache.clear()

    def get_tracking_manager(self, camera_id):
        with self.camera_lock:
            runtime = self.cameras.get(camera_id)
        return runtime.tracking_manager if runtime else None

    # kiểm tra xem còn camera nào đang chờ kết quả plate/OCR hay không
    # được PipelineManager sử dụng trước khi shutdown
    def has_pending_work(self):
        with self.camera_lock:
            return any(
                runtime.pending_plate or runtime.pending_ocr
                for runtime in self.cameras.values()
            )

    # Chạy liên tục cho đến khi PipelineManager.stop() gọi self.state.stop_event.set().
    def run(self):
        while self.state.running:
            # Ưu tiên xử lý kết quả đã trả về từ các worker inference.
            # các kq nằm trong result_queue
            handled = self._consume_result()
            # Nếu chưa có result thì lấy frame mới từ các camera queue và xử lý tiếp.
            handled = self._consume_frame() or handled
            if not handled:
                # Nếu không có gì để làm thì ngủ rất ngắn 10ms để tránh busy-spin ăn CPU.
                self.state.stop_event.wait(0.01)

        self._drain_results()
        self._finalize_all()

    # Lấy 1 item từ result_queue
    # Nếu là VehicleResult thì _handle_vehicle_result()
    # Nếu là PlateResult/OCRResult thì merge vào track tương ứng.
    # Mục tiêu: cập nhật state sau khi inference xong.
    def _consume_result(self):
        try:
            result = self.state.result_queue.get_nowait()
        except Empty:
            return False

        try:
            try:
                if isinstance(result, VehicleResult):
                    self._handle_vehicle_result(result)
                elif isinstance(result, OCRResult):
                    self._handle_ocr_result(result)
                elif isinstance(result, PlateResult):
                    self._handle_plate_result(result)
            except Exception as exc:
                self.state.report_error(
                    self.name,
                    getattr(result, "camera_id", None),
                    exc,
                )
        finally:
            self.state.result_queue.task_done()
        return True

    # Mục tiêu: tạo nhịp giữa nhiều camera.
    # Quét các camera theo round-robin
    # Lấy 1 frame từ queue camera nào có dữ liệu
    def _consume_frame(self):
        with self.camera_lock:
            camera_order = list(self.camera_order)
            camera_cursor = self.camera_cursor

        if not camera_order:
            return False

        packet = None
        source_queue = None
        camera_count = len(camera_order)
        # Duyệt qua toàn bộ camera
        for offset in range(camera_count):
            # Chọn camera theo thứ tự luân phiên 1 -> 3 -> 1
            index = (camera_cursor + offset) % camera_count
            camera_id = camera_order[index]
            snapshot = self.state.get_camera(camera_id)
            if snapshot is None:
                continue
            try:
                # Lấy frame mới nhất của camera đó mà không chờ block
                packet = snapshot.frame_queue.get_nowait()
                source_queue = snapshot.frame_queue
                with self.camera_lock:
                    if camera_id in self.camera_order:
                        self.camera_cursor = (
                            self.camera_order.index(camera_id) + 1
                        ) % len(self.camera_order)
                break
            except Empty:
                continue

        if packet is None:
            return False

        try:
            try:
                self._handle_frame(packet)
            except Exception as exc:
                self.state.report_error(
                    self.name,
                    packet.camera_id,
                    exc,
                )
        finally:
            source_queue.task_done()
        return True

    # Khi hệ thống sắp dừng, hàm này:
    # hút sạch result_queue, xử lý hết result còn tồn
    # tránh mất kết quả OCR/vehicle đã chạy xong nhưng chưa được merge vào track
    def _drain_results(self):
        while True:
            try:
                result = self.state.result_queue.get_nowait()
            except Empty:
                return
            try:
                try:
                    if isinstance(result, VehicleResult):
                        self._handle_vehicle_result(result)
                    elif isinstance(result, OCRResult):
                        self._handle_ocr_result(result)
                    elif isinstance(result, PlateResult):
                        self._handle_plate_result(result)
                except Exception as exc:
                    self.state.report_error(
                        self.name,
                        getattr(result, "camera_id", None),
                        exc,
                    )
            finally:
                self.state.result_queue.task_done()

    # xử lý một frame vừa lấy từ queue của camera.
    def _handle_frame(self, packet: FrameData):
        runtime = self.cameras.get(packet.camera_id)
        snapshot = self.state.get_camera(packet.camera_id)
        if runtime is None or snapshot is None:
            return

        self.state.record_stat(packet.camera_id, "frame_consumed")

        vehicle_task_queued = False
        if packet.frame_id % VEHICLE_STEP == 0:
            task = VehicleTask(
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                frame=packet.frame.copy(),
            )
            if not put_latest(self.state.vehicle_task_queue, task):
                self.state.record_stat(packet.camera_id, "vehicle_task_dropped")
            else:
                self.state.record_stat(packet.camera_id, "vehicle_tasks")
                vehicle_task_queued = True

        if not vehicle_task_queued:
            self._publish_snapshot(packet.camera_id, packet.frame, packet.frame_id)

    def _handle_vehicle_result(self, result: VehicleResult):
        runtime = self.cameras.get(result.camera_id)
        if runtime is None:
            return

        # if result.frame_id <= runtime.last_vehicle_result_frame:
        #     return
        runtime.last_vehicle_result_frame = result.frame_id

        vehicles = runtime.tracker.update(result.detections)
        runtime.latest_vehicles = vehicles
        runtime.frame_cache[result.frame_id] = result.frame.copy()
        active_track_ids = {vehicle[0] for vehicle in vehicles}
        self.state.record_stat(result.camera_id, "vehicle_results")

        self._update_tracks(runtime, result.frame_id, result.frame, vehicles)
        self._schedule_plate(result.camera_id, runtime, result.frame_id)
        self._schedule_ocr(result.camera_id, runtime, result.frame_id)
        self._update_missing_tracks(runtime, result.frame_id, active_track_ids)
        runtime.tracking_manager.remove_expired_tracks(result.frame_id)
        self._prune_frame_cache(runtime)
        self._publish_snapshot(
            result.camera_id,
            result.frame,
            result.frame_id,
            force=True,
        )

    def _schedule_plate(self, camera_id, runtime, frame_id):
        scheduled = 0
        manager = runtime.tracking_manager
        for track_id, x1, y1, x2, y2, confidence, label in runtime.latest_vehicles:
            if scheduled >= MAX_PLATE_FRAME_CHECKS:
                break

            memory = manager.memory.get(track_id)
            if memory is None:
                continue
            if memory.get("best_text") and memory.get("best_count", 0) >= manager.min_votes:
                continue
            # if not manager.can_run_anpr(track_id, frame_id):
            #     continue
            if track_id in runtime.pending_plate:
                continue

            samples = manager.get_best_vehicle_samples(track_id, top_k=3)
            if not samples:
                continue

            has_unattempted_sample = any(
                sample.get("plate_attempts", 0) == 0
                and not sample.get("plate_checked")
                for sample in samples
            )
            if frame_id % PLATE_STEP != 0 and not has_unattempted_sample:
                continue

            candidates = []
            for sample in samples:
                if sample.get("plate_checked"):
                    continue

                attempts = sample.get("plate_attempts", 0)
                if attempts >= MAX_PLATE_SAMPLE_RETRIES:
                    sample["plate_checked"] = True
                    continue
                if frame_id < sample.get("retry_after_frame", -1):
                    continue

                frame_ref = sample.get("frame_ref", sample.get("frame_idx"))
                scene_frame = runtime.frame_cache.get(frame_ref)
                if scene_frame is not None and scene_frame.size > 0:
                    candidates.append(
                        (
                            attempts,
                            -sample.get("priority", 0),
                            sample,
                            scene_frame,
                        )
                    )

            if not candidates:
                continue

            _, _, selected_sample, scene_frame = min(
                candidates,
                key=lambda candidate: (candidate[0], candidate[1]),
            )
            source_frame_id = selected_sample.get(
                "frame_ref",
                selected_sample.get("frame_idx"),
            )
            task = PlateTask(
                camera_id=camera_id,
                frame_id=frame_id,
                source_frame_id=source_frame_id,
                track_id=track_id,
                track_started_at=memory["first_seen_frame"],
                vehicle_bbox=selected_sample.get("bbox", (x1, y1, x2, y2)),
                vehicle_label=memory.get("vehicle_type", label),
                frame=scene_frame.copy(),
            )
            if put_if_available(self.state.plate_task_queue, task):
                self.state.record_stat(camera_id, "plate_tasks")
                selected_sample["plate_attempts"] = (
                    selected_sample.get("plate_attempts", 0) + 1
                )
                selected_sample["retry_after_frame"] = frame_id + PLATE_STEP
                runtime.pending_plate.add(track_id)
                scheduled += 1
            else:
                self.state.record_stat(camera_id, "plate_task_dropped")

    def _schedule_ocr(self, camera_id, runtime, frame_id, force=False, track_ids=None):
        if not force and frame_id % OCR_STEP != 0:
            return

        manager = runtime.tracking_manager
        active_tracks = (
            list(track_ids)
            if track_ids is not None
            else [vehicle[0] for vehicle in runtime.latest_vehicles]
        )
        scheduled = 0

        for track_id in active_tracks:
            if scheduled >= MAX_OCR_READS:
                break

            memory = manager.memory.get(track_id)
            if memory is None:
                continue
            if memory.get("best_text") and memory.get("best_count", 0) >= manager.min_votes:
                continue

            samples = manager.get_best_plate_samples(track_id, top_k=3)
            if not samples:
                continue

            for sample in samples:
                if scheduled >= MAX_OCR_READS:
                    break
                if (
                    sample.get("ocr_success")
                    or sample.get("ocr_failed")
                    or sample.get("ocr_pending")
                ):
                    continue

                plate_img = sample.get("plate_img")
                if plate_img is None or plate_img.size == 0:
                    sample["ocr_failed"] = True
                    continue

                ocr_key = self._ocr_sample_key(track_id, sample)
                if ocr_key in runtime.pending_ocr:
                    continue

                task = CharTask(
                    camera_id=camera_id,
                    frame_id=frame_id,
                    source_frame_id=sample.get("frame_idx"),
                    track_id=track_id,
                    track_started_at=memory["first_seen_frame"],
                    vehicle_bbox=sample.get("vehicle_bbox") or memory.get("vehicle_bbox"),
                    vehicle_label=memory.get("vehicle_type", "unknown"),
                    plate_bbox=sample.get("bbox"),
                    plate_points=sample.get("pts"),
                    plate_img=plate_img.copy(),
                    sharpness=sample.get("sharpness", 0.0),
                )
                if put_if_available(self.state.char_task_queue, task):
                    sample["ocr_pending"] = True
                    sample["ocr_attempts"] = sample.get("ocr_attempts", 0) + 1
                    runtime.pending_ocr.add(ocr_key)
                    self.state.record_stat(camera_id, "char_tasks")
                    scheduled += 1
                else:
                    self.state.record_stat(camera_id, "char_task_dropped")

    def _handle_plate_result(self, result: PlateResult):
        runtime = self.cameras.get(result.camera_id)
        if runtime is None:
            return

        runtime.pending_plate.discard(result.track_id)
        manager = runtime.tracking_manager
        memory = manager.memory.get(result.track_id)
        if memory is None:
            return
        if memory.get("first_seen_frame") != result.track_started_at:
            return

        source_sample = self._find_vehicle_sample(
            memory,
            result.source_frame_id,
            result.vehicle_bbox,
        )
        if source_sample is not None:
            source_sample["last_plate_reason"] = result.reason
            source_sample["last_plate_frame"] = result.frame_id

        if result.plate_bbox is not None and result.plate_img is not None:
            memory["last_plate_view"] = {
                "frame_id": result.source_frame_id,
                "vehicle_bbox": result.vehicle_bbox,
                "plate_bbox": result.plate_bbox,
                "plate_points": (
                    np.asarray(result.plate_points).copy()
                    if result.plate_points is not None
                    else None
                ),
                "plate_img": result.plate_img.copy(),
                "text": "",
                "reason": result.reason,
                "success": result.success,
                "avg_confidence": 0.0,
            }

        if result.success and result.plate_bbox is not None:
            bbox = result.plate_bbox
            area = max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            priority = result.sharpness + 0.01 * area
            manager.update_best_plate_frame(
                track_id=result.track_id,
                frame_index=result.source_frame_id,
                bbox=bbox,
                sharpness=result.sharpness,
                priority=priority,
                plate_img=result.plate_img,
                pts=result.plate_points,
                vehicle_bbox=result.vehicle_bbox,
                max_samples=8,
            )
            if source_sample is not None:
                source_sample["plate_checked"] = True
            self.state.record_stat(result.camera_id, "plate_results")
            self._schedule_ocr(
                result.camera_id,
                runtime,
                result.frame_id,
                force=True,
                track_ids=[result.track_id],
            )
        elif result.reason == "blur_or_cut":
            if source_sample is not None:
                source_sample["plate_checked"] = (
                    source_sample.get("plate_attempts", 0)
                    >= MAX_PLATE_SAMPLE_RETRIES
                )
            manager.update_unreadable_plate(
                result.track_id,
                result.frame_id,
                result.sharpness,
                result.reason,
            )
            manager.update_after_anpr(
                result.track_id,
                result.frame_id,
                result.sharpness,
            )
            self.state.record_stat(result.camera_id, "plate_results")
        elif result.reason in {"no_plate", "invalid_vehicle_bbox"}:
            if source_sample is not None:
                source_sample["plate_checked"] = (
                    source_sample.get("plate_attempts", 0)
                    >= MAX_PLATE_SAMPLE_RETRIES
                )
            manager.update_no_plate(result.track_id, result.frame_id)
            manager.update_after_anpr(result.track_id, result.frame_id, 0.0)
            self.state.record_stat(result.camera_id, "plate_results")

        snapshot = self.state.get_camera(result.camera_id)
        if snapshot is not None:
            with snapshot.lock:
                frame = (
                    snapshot.latest_frame.copy()
                    if snapshot.latest_frame is not None
                    else None
                )
            if frame is not None:
                self._publish_snapshot(result.camera_id, frame)

    def _handle_ocr_result(self, result: OCRResult):
        runtime = self.cameras.get(result.camera_id)
        if runtime is None:
            return

        runtime.pending_ocr.discard(self._ocr_result_key(result))
        manager = runtime.tracking_manager
        memory = manager.memory.get(result.track_id)
        if memory is None:
            return
        if memory.get("first_seen_frame") != result.track_started_at:
            return

        plate_sample = self._find_plate_sample(
            memory,
            result.source_frame_id,
            result.vehicle_bbox,
            result.plate_bbox,
        )
        if plate_sample is not None:
            plate_sample["ocr_pending"] = False
            plate_sample["last_ocr_reason"] = result.reason
            plate_sample["last_ocr_frame"] = result.frame_id

        if result.plate_bbox is not None and result.plate_img is not None:
            memory["last_plate_view"] = {
                "frame_id": result.source_frame_id,
                "vehicle_bbox": result.vehicle_bbox,
                "plate_bbox": result.plate_bbox,
                "plate_points": (
                    np.asarray(result.plate_points).copy()
                    if result.plate_points is not None
                    else None
                ),
                "plate_img": result.plate_img.copy(),
                "text": result.text,
                "reason": result.reason,
                "success": result.success,
                "avg_confidence": result.avg_confidence,
            }

        if result.success and result.plate_bbox is not None:
            successful_reads_before = memory.get("successful_reads", 0)
            manager.update_plate(
                track_id=result.track_id,
                text=result.text,
                frame_index=result.frame_id,
                sharpness=result.sharpness,
                char_count=result.char_count,
                avg_conf=result.avg_confidence,
                bbox=result.plate_bbox,
            )
            accepted_read = (
                memory.get("successful_reads", 0) > successful_reads_before
            )
            if plate_sample is not None:
                if accepted_read:
                    plate_sample["ocr_success"] = True
                    plate_sample["ocr_text"] = result.text
                    plate_sample["ocr_confidence"] = result.avg_confidence
                    plate_sample["ocr_score"] = (
                        result.sharpness
                        + result.avg_confidence * 50.0
                        + result.char_count * 2.0
                    )
                else:
                    plate_sample["ocr_failed"] = True
            quality = result.sharpness + result.avg_confidence * 50.0
            manager.update_after_anpr(result.track_id, result.frame_id, quality)
            manager.update_display_info(
                result.track_id,
                manager.get_stable_text(result.track_id) or result.text,
            )
            self.state.record_stat(result.camera_id, "char_results")
        elif result.reason in {
            "blur_or_cut",
            "insufficient_chars",
            "invalid_text",
            "invalid_plate_img",
        }:
            if plate_sample is not None:
                plate_sample["ocr_failed"] = True
            manager.update_unreadable_plate(
                result.track_id,
                result.frame_id,
                result.sharpness,
                result.reason,
            )
            manager.update_after_anpr(
                result.track_id,
                result.frame_id,
                result.sharpness,
            )
            self.state.record_stat(result.camera_id, "char_results")

        if not (
            memory.get("best_text")
            and memory.get("best_count", 0) >= manager.min_votes
        ):
            self._schedule_ocr(
                result.camera_id,
                runtime,
                result.frame_id,
                force=True,
                track_ids=[result.track_id],
            )

        snapshot = self.state.get_camera(result.camera_id)
        if snapshot is not None:
            with snapshot.lock:
                frame = (
                    snapshot.latest_frame.copy()
                    if snapshot.latest_frame is not None
                    else None
                )
            if frame is not None:
                self._publish_snapshot(result.camera_id, frame)

    @staticmethod
    def _normalize_bbox(bbox):
        if bbox is None:
            return None
        return tuple(int(value) for value in bbox)

    @classmethod
    def _ocr_sample_key(cls, track_id, sample):
        return (
            track_id,
            sample.get("frame_idx"),
            cls._normalize_bbox(sample.get("vehicle_bbox")),
            cls._normalize_bbox(sample.get("bbox")),
        )

    @classmethod
    def _ocr_result_key(cls, result):
        return (
            result.track_id,
            result.source_frame_id,
            cls._normalize_bbox(result.vehicle_bbox),
            cls._normalize_bbox(result.plate_bbox),
        )

    @staticmethod
    def _find_vehicle_sample(memory, frame_id, vehicle_bbox):
        for sample in memory.get("best_vehicle_frames", []):
            if (
                sample.get("frame_ref", sample.get("frame_idx")) == frame_id
                and sample.get("bbox") == vehicle_bbox
            ):
                return sample
        for sample in memory.get("best_vehicle_frames", []):
            if sample.get("frame_ref", sample.get("frame_idx")) == frame_id:
                return sample
        return None

    @staticmethod
    def _find_plate_sample(memory, frame_id, vehicle_bbox, plate_bbox=None):
        for sample in memory.get("best_plate_frames", []):
            if (
                sample.get("frame_idx") == frame_id
                and sample.get("vehicle_bbox") == vehicle_bbox
                and (plate_bbox is None or sample.get("bbox") == plate_bbox)
            ):
                return sample
        for sample in memory.get("best_plate_frames", []):
            if (
                sample.get("frame_idx") == frame_id
                and sample.get("vehicle_bbox") == vehicle_bbox
            ):
                return sample
        return None

    @staticmethod
    def _expand_vehicle_sample_bbox(bbox, frame_width, frame_height, label):
        x1, y1, x2, y2 = bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)

        pad_x = int(box_width * 0.10)
        pad_top = int(box_height * 0.06)
        pad_bottom_ratio = 0.24 if label == "motorbike" else 0.18
        pad_bottom = int(box_height * pad_bottom_ratio)

        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_top),
            min(frame_width, x2 + pad_x),
            min(frame_height, y2 + pad_bottom),
        )

    @staticmethod
    def _edge_crop_penalty(bbox, frame_width, frame_height):
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        touched_edges = 0
        if x1 <= 0:
            touched_edges += 1
        if y1 <= 0:
            touched_edges += 1
        if x2 >= frame_width:
            touched_edges += 1
        if y2 >= frame_height:
            touched_edges += 1
        return touched_edges * area * 0.004

    def _update_tracks(self, runtime, frame_id, frame, vehicles):
        height, width = frame.shape[:2]
        manager = runtime.tracking_manager
 
        if frame_id % VEHICLE_SAMPLE_STEP == 0 and vehicles: # VEHICLE_SAMPLE_STEP
            runtime.frame_cache[frame_id] = frame.copy()

        for track_id, x1, y1, x2, y2, confidence, label in vehicles:
            if track_id not in manager.memory:
                manager.init_track(
                    track_id,
                    vehicle_type=label,
                    frame_index=frame_id,
                )

            memory = manager.memory[track_id]
            stable_label = manager.update_vehicle_type(
                track_id,
                label,
                confidence,
            )
            memory["vehicle_bbox"] = (x1, y1, x2, y2)
            memory["last_seen_frame"] = frame_id
            memory["last_seen_time"] = datetime.now().isoformat()
            memory["missing_frames"] = 0

            if frame_id % VEHICLE_SAMPLE_STEP != 0: # VEHICLE_SAMPLE_STEP
                continue

            x1c, y1c, x2c, y2c = self._expand_vehicle_sample_bbox(
                (x1, y1, x2, y2),
                width,
                height,
                stable_label,
            )
            # x1c, y1c = max(0, x1), max(0, y1)
            # x2c, y2c = min(width, x2), min(height, y2)
            # if x2c <= x1c or y2c <= y1c:
            #     continue

            vehicle_img = frame[y1c:y2c, x1c:x2c]
            if vehicle_img.size == 0:
                continue

            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            area = (x2c - x1c) * (y2c - y1c)
            edge_penalty = self._edge_crop_penalty(
                (x1c, y1c, x2c, y2c),
                width,
                height,
            )
            manager.update_best_vehicle_frame(
                track_id=track_id,
                frame_index=frame_id,
                bbox=(x1c, y1c, x2c, y2c),
                sharpness=sharpness,
                priority=sharpness + 0.01 * area - edge_penalty,
                vehicle_img=vehicle_img.copy(),
                frame_ref=frame_id,
                max_samples=6,
            )

    def _update_missing_tracks(self, runtime, frame_id, active_track_ids):
        manager = runtime.tracking_manager
        for track_id in list(manager.memory.keys()):
            if track_id in active_track_ids:
                continue

            memory = manager.memory[track_id]
            memory["missing_frames"] = memory.get("missing_frames", 0) + 1
            if memory["missing_frames"] < TRACK_LOST_APPEND_FRAMES:
                continue

            manager.finalize_track(track_id)
            manager.memory.pop(track_id, None)
            runtime.pending_plate.discard(track_id)
            runtime.pending_ocr = {
                key for key in runtime.pending_ocr
                if not key or key[0] != track_id
            }

    # dọn các frame cũ trong cache để tiết kiệm RAM.
    @staticmethod
    def _prune_frame_cache(runtime):
        # Tạo tập frame cần giữ.
        keep_refs = set()
        for memory in runtime.tracking_manager.memory.values():
            # Giữ lại những frame đã được chọn làm frame xe tốt.
            for sample in memory.get("best_vehicle_frames", []):
                frame_ref = sample.get("frame_ref")
                if frame_ref is not None:
                    keep_refs.add(frame_ref)

        # Ngoài frame đáng giữ, cũng giữ thêm vài frame mới nhất để hỗ trợ retry ANPR.
        newest = sorted(runtime.frame_cache.keys(), reverse=True)[
            :MAX_CACHED_FRAMES
        ]
        keep_refs.update(newest)
        # Xóa tất cả frame không còn cần thiết.
        for frame_ref in list(runtime.frame_cache.keys()):
            if frame_ref not in keep_refs:
                runtime.frame_cache.pop(frame_ref, None)

    # tạo ảnh render và đẩy kết quả ra ngoài.
    def _publish_snapshot(self, camera_id, frame=None, frame_id=None, force=False):
        runtime = self.cameras[camera_id]
        snapshot = self.state.get_camera(camera_id)
        if snapshot is None:
            return

        with snapshot.lock:
            if frame is not None and frame_id is not None:
                base_frame = frame.copy()
                base_frame_id = frame_id
            elif snapshot.latest_frame is not None:
                base_frame = snapshot.latest_frame.copy()
                base_frame_id = snapshot.latest_frame_id
            elif frame is not None:
                base_frame = frame.copy()
                base_frame_id = -1
            else:
                return

            is_old_frame = base_frame_id < snapshot.latest_render_frame_id
            is_same_frame = base_frame_id == snapshot.latest_render_frame_id
            if is_old_frame or (is_same_frame and not force):
                snapshot.latest_results = (
                    runtime.tracking_manager.export_all_results()
                )
                return

        draw_vehicle_boxes = base_frame_id == runtime.last_vehicle_result_frame
        rendered = self._render(
            base_frame,
            runtime,
            draw_vehicle_boxes=draw_vehicle_boxes,
        )
        results = runtime.tracking_manager.export_all_results()
        with snapshot.lock:
            snapshot.latest_render_frame = rendered
            snapshot.latest_render_frame_id = base_frame_id
            snapshot.latest_results = results
        self.state.record_stat(camera_id, "renders")

        put_latest(
            self.state.output_queue,
            (camera_id, rendered, results),
        )

        if self.on_results is not None:
            self.on_results(camera_id, rendered, results)

    @staticmethod
    def _map_point(point, source_bbox, target_bbox):
        sx1, sy1, sx2, sy2 = source_bbox
        tx1, ty1, tx2, ty2 = target_bbox
        source_w = max(1, sx2 - sx1)
        source_h = max(1, sy2 - sy1)
        scale_x = (tx2 - tx1) / source_w
        scale_y = (ty2 - ty1) / source_h
        return (
            int(tx1 + (point[0] - sx1) * scale_x),
            int(ty1 + (point[1] - sy1) * scale_y),
        )

    @classmethod
    def _draw_plate_debug(cls, frame, vehicle_bbox, plate_view, text):
        plate_img = plate_view.get("plate_img")
        source_vehicle_bbox = plate_view.get("vehicle_bbox")
        plate_bbox = plate_view.get("plate_bbox")
        if (
            plate_img is None
            or plate_img.size == 0
            or source_vehicle_bbox is None
            or plate_bbox is None
        ):
            return

        height, width = frame.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_bbox
        vehicle_w = max(1, vx2 - vx1)
        vehicle_h = max(1, vy2 - vy1)

        mapped_plate = [
            cls._map_point((plate_bbox[0], plate_bbox[1]), source_vehicle_bbox, vehicle_bbox),
            cls._map_point((plate_bbox[2], plate_bbox[3]), source_vehicle_bbox, vehicle_bbox),
        ]
        # cv2.rectangle(frame, mapped_plate[0], mapped_plate[1], (0, 0, 255), 2)

        preview_w = max(60, int(vehicle_w * 0.45))
        aspect = plate_img.shape[0] / max(1, plate_img.shape[1])
        preview_h = max(20, int(preview_w * aspect))
        max_preview_h = max(20, int(vehicle_h * 0.35))
        if preview_h > max_preview_h:
            preview_h = max_preview_h
            preview_w = max(1, int(preview_h / max(aspect, 1e-6)))

        margin = 8
        draw_x1 = max(0, min(width - 1, vx2 - preview_w - margin))
        draw_y1 = max(0, min(height - 1, vy1 + margin))
        draw_x2 = min(width, draw_x1 + preview_w)
        draw_y2 = min(height, draw_y1 + preview_h)
        if draw_x2 <= draw_x1 or draw_y2 <= draw_y1:
            return

        preview = cv2.resize(
            plate_img,
            (draw_x2 - draw_x1, draw_y2 - draw_y1),
        )
        frame[draw_y1:draw_y2, draw_x1:draw_x2] = preview
        cv2.rectangle(
            frame,
            (draw_x1, draw_y1),
            (draw_x2, draw_y2),
            (0, 0, 255),
            2,
        )

        debug_text = text or plate_view.get("text", "")
        if not debug_text:
            debug_text = plate_view.get("reason", "")
        if debug_text:
            text_y = min(height - 5, draw_y2 + 22)
            cv2.putText(
                frame,
                debug_text,
                (draw_x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if plate_view.get("success") else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    @classmethod
    def _render(cls, frame, runtime, draw_vehicle_boxes=True):
        colors = {
            "oto": (255, 0, 0),
            "xe-tai": (0, 255, 255),
            "xe-container": (0, 165, 255),
            "motorbike": (0, 255, 0),
        }
        if not draw_vehicle_boxes:
            return frame

        for track_id, x1, y1, x2, y2, confidence, label in runtime.latest_vehicles:
            memory = runtime.tracking_manager.memory.get(track_id)
            display_label = (
                memory.get("vehicle_type", label)
                if memory is not None
                else label
            )
            color = colors.get(display_label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = runtime.tracking_manager.get_stable_text(track_id)
            display_info = runtime.tracking_manager.get_display_info(track_id)
            if not text and display_info is not None:
                text = display_info.get("text", "")
            caption = f"ID:{track_id} {display_label} {confidence:.2f}"
            if text:
                caption += f" {text}"
            text_origin = (x1, max(20, y1 - 8))
            text_size, baseline = cv2.getTextSize(
                caption,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            cv2.rectangle(
                frame,
                (text_origin[0], text_origin[1] - text_size[1] - baseline),
                (text_origin[0] + text_size[0], text_origin[1] + baseline),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                caption,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            if memory is not None:
                plate_view = memory.get("last_plate_view")
                if plate_view:
                    cls._draw_plate_debug(
                        frame,
                        (x1, y1, x2, y2),
                        plate_view,
                        text,
                    )
        return frame

    def _finalize_all(self):
        with self.camera_lock:
            camera_items = list(self.cameras.items())

        for camera_id, runtime in camera_items:
            runtime.tracking_manager.finalize_all_active_tracks()
            snapshot = self.state.get_camera(camera_id)
            if snapshot is not None:
                with snapshot.lock:
                    snapshot.latest_results = (
                        runtime.tracking_manager.export_all_results()
                    )