import threading
from queue import Empty

from src.config.settings import (
    CHAR_SHARPNESS_THRESHOLD,
    PLATE_SHARPNESS_THRESHOLD,
)
from src.main.Filter_Plate import Filter_Plate
from src.main.PlateWarper import PlateWarper
from src.pipeline.ANPRUtils import chars_to_text, format_plate
from src.pipeline.ResultTypes import OCRResult, PlateResult, VehicleResult
from src.pipeline.SharedsState import put_until_stopped


class VehicleInferenceWorker(threading.Thread):
    def __init__(self, state, vehicle_model):
        super().__init__(name="vehicle-inference", daemon=True)
        self.state = state
        self.vehicle_model = vehicle_model

    def run(self):
        while self.state.running:
            try:
                task = self.state.vehicle_task_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                detections = self.vehicle_model.detect(task.frame)
                result = VehicleResult(
                    camera_id=task.camera_id,
                    frame_id=task.frame_id,
                    timestamp=task.timestamp,
                    frame=task.frame,
                    detections=detections or [],
                )
                put_until_stopped(
                    self.state.result_queue,
                    result,
                    self.state.stop_event,
                )
            except Exception as exc:
                self.state.report_error(
                    self.name,
                    task.camera_id,
                    exc,
                )
            finally:
                self.state.vehicle_task_queue.task_done()


class PlateInferenceWorker(threading.Thread):
    def __init__(self, state, plate_model):
        super().__init__(name="plate-inference", daemon=True)
        self.state = state
        self.plate_model = plate_model
        self.warper = PlateWarper(
            sharpness_threshold=PLATE_SHARPNESS_THRESHOLD,
            sharpness_method="laplacian",
        )
        self.filter_plate = Filter_Plate(
            method="laplacian",
            char_threshold=CHAR_SHARPNESS_THRESHOLD,
            plate_threshold=PLATE_SHARPNESS_THRESHOLD,
        )

    def run(self):
        while self.state.running:
            try:
                task = self.state.plate_task_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                result = self._process(task)
                put_until_stopped(
                    self.state.result_queue,
                    result,
                    self.state.stop_event,
                )
            except Exception as exc:
                self.state.report_error(
                    self.name,
                    task.camera_id,
                    exc,
                )
            finally:
                self.state.plate_task_queue.task_done()

    def _process(self, task):
        frame_height, frame_width = task.frame.shape[:2]
        vx1, vy1, vx2, vy2 = task.vehicle_bbox
        vx1 = max(0, min(frame_width, vx1))
        vx2 = max(0, min(frame_width, vx2))
        vy1 = max(0, min(frame_height, vy1))
        vy2 = max(0, min(frame_height, vy2))

        if vx2 <= vx1 or vy2 <= vy1:
            return self._failure(task, "invalid_vehicle_bbox")

        plates = self.plate_model.detect(task.frame)
        if not plates:
            return self._failure(task, "no_plate")

        best_result = None
        best_priority = float("-inf")

        for plate in plates:
            local_bbox = plate["bbox"]
            local_points = plate.get("points")

            px1, py1, px2, py2 = local_bbox
            cx = (px1 + px2) / 2.0
            cy = (py1 + py2) / 2.0
            if not (vx1 <= cx <= vx2 and vy1 <= cy <= vy2):
                continue

            global_bbox = local_bbox
            global_points = local_points.copy() if local_points is not None else None

            plate_img, sharpness, is_usable = self._prepare_plate(
                task.frame,
                global_bbox,
                global_points,
            )

            if not is_usable:
                candidate = self._failure(
                    task,
                    "blur_or_cut",
                    plate_bbox=global_bbox,
                    plate_points=global_points,
                    plate_img=plate_img,
                    sharpness=sharpness,
                )
            else:
                candidate = PlateResult(
                    camera_id=task.camera_id,
                    frame_id=task.frame_id,
                    source_frame_id=task.source_frame_id,
                    track_id=task.track_id,
                    track_started_at=task.track_started_at,
                    success=True,
                    reason="ok",
                    vehicle_bbox=task.vehicle_bbox,
                    vehicle_label=task.vehicle_label,
                    plate_bbox=global_bbox,
                    plate_points=global_points,
                    plate_img=plate_img,
                    sharpness=sharpness,
                )

            area = max(
                0,
                (global_bbox[2] - global_bbox[0])
                * (global_bbox[3] - global_bbox[1]),
            )
            priority = sharpness + 0.01 * area
            if candidate.success:
                priority += 10000
            if priority > best_priority:
                best_priority = priority
                best_result = candidate

        return best_result or self._failure(task, "no_plate")

    def _prepare_plate(self, frame, bbox, points):
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        plate_crop = frame[y1:y2, x1:x2].copy()

        if plate_crop.size == 0:
            return plate_crop, 0.0, False

        plate_img = plate_crop
        if points is not None and self.warper.is_valid_plate(points):
            warped = self.warper.warp(frame, points)
            if warped is not None and warped.size > 0:
                plate_img = warped

        sharpness = self.filter_plate.measure_sharpness(plate_img)
        return (
            plate_img,
            sharpness,
            sharpness >= self.filter_plate.plate_threshold,
        )

    def _failure(
        self,
        task,
        reason,
        plate_bbox=None,
        plate_points=None,
        plate_img=None,
        sharpness=0.0,
    ):
        return PlateResult(
            camera_id=task.camera_id,
            frame_id=task.frame_id,
            source_frame_id=task.source_frame_id,
            track_id=task.track_id,
            track_started_at=task.track_started_at,
            success=False,
            reason=reason,
            vehicle_bbox=task.vehicle_bbox,
            vehicle_label=task.vehicle_label,
            plate_bbox=plate_bbox,
            plate_points=plate_points,
            plate_img=plate_img,
            sharpness=sharpness,
        )


class OCRInferenceWorker(threading.Thread):
    def __init__(self, state, char_model):
        super().__init__(name="ocr-inference", daemon=True)
        self.state = state
        self.char_model = char_model
        self.filter_plate = Filter_Plate(
            method="laplacian",
            char_threshold=CHAR_SHARPNESS_THRESHOLD,
            plate_threshold=PLATE_SHARPNESS_THRESHOLD,
        )

    def run(self):
        while self.state.running:
            try:
                task = self.state.char_task_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                result = self._process(task)
                put_until_stopped(
                    self.state.result_queue,
                    result,
                    self.state.stop_event,
                )
            except Exception as exc:
                self.state.report_error(
                    self.name,
                    task.camera_id,
                    exc,
                )
            finally:
                self.state.char_task_queue.task_done()

    def _process(self, task):
        plate_img = task.plate_img
        if plate_img is None or plate_img.size == 0:
            return self._failure(task, "invalid_plate_img")

        chars = self.char_model.detect(plate_img)
        chars = self.filter_plate.char_sharpness(plate_img, chars or [])
        if len(chars) < 4:
            return self._failure(
                task,
                "insufficient_chars",
            )

        text = format_plate(chars_to_text(chars))
        confidences = [char.get("conf", 0.0) for char in chars]
        average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return OCRResult(
            camera_id=task.camera_id,
            frame_id=task.frame_id,
            source_frame_id=task.source_frame_id,
            track_id=task.track_id,
            track_started_at=task.track_started_at,
            success=bool(text),
            reason="ok" if text else "invalid_text",
            vehicle_bbox=task.vehicle_bbox,
            vehicle_label=task.vehicle_label,
            plate_bbox=task.plate_bbox,
            plate_points=task.plate_points,
            plate_img=plate_img,
            sharpness=task.sharpness,
            text=text,
            avg_confidence=average_confidence,
            char_count=len(chars),
        )

    def _failure(
        self,
        task,
        reason,
    ):
        return OCRResult(
            camera_id=task.camera_id,
            frame_id=task.frame_id,
            source_frame_id=task.source_frame_id,
            track_id=task.track_id,
            track_started_at=task.track_started_at,
            success=False,
            reason=reason,
            vehicle_bbox=task.vehicle_bbox,
            vehicle_label=task.vehicle_label,
            plate_bbox=task.plate_bbox,
            plate_points=task.plate_points,
            plate_img=task.plate_img,
            sharpness=task.sharpness,
        )
