import time
import threading
from queue import Empty

from src.pipeline.CaptureWorker import CaptureWorker
from src.pipeline.InferenceWorker import (
    OCRInferenceWorker,
    PlateInferenceWorker,
    VehicleInferenceWorker,
)
from src.pipeline.RenderWorker import RenderWorker
from src.pipeline.SharedsState import SharedState
from src.pipeline.TrackingWorker import TrackingWorker


class PipelineManager:
    def __init__(
        self,
        cap=None,
        vehicle_model=None,
        plate_model=None,
        char_model=None,
        tracking_manager=None,
        show=False,
        on_results=None,
        event_publisher=None,
        tracker_factory=None,
    ):
        if vehicle_model is None or plate_model is None or char_model is None:
            raise ValueError("vehicle_model, plate_model and char_model are required")

        self.state = SharedState()
        self.show = show
        self.capture_workers = {}
        self.started = False
        self.lifecycle_lock = threading.RLock()
        self.event_publisher = event_publisher
        self.event_publisher_closed = False

        def publish_results(camera_id, frame, results):
            if on_results is not None:
                on_results(camera_id, frame, results)
            if self.event_publisher is not None:
                self.event_publisher(camera_id, frame, results)

        if tracking_manager is not None:
            used = False

            def tracking_manager_factory():
                nonlocal used
                if used:
                    raise ValueError(
                        "A TrackingManager instance can only serve one camera"
                    )
                used = True
                return tracking_manager
        else:
            tracking_manager_factory = None

        self.tracking_worker = TrackingWorker(
            self.state,
            tracker_factory=tracker_factory,
            tracking_manager_factory=tracking_manager_factory,
            on_results=publish_results,
        )
        self.vehicle_worker = VehicleInferenceWorker(
            self.state,
            vehicle_model,
        )
        self.plate_worker = PlateInferenceWorker(
            self.state,
            plate_model,
        )
        self.ocr_worker = OCRInferenceWorker(
            self.state,
            char_model,
        )
        self.render_worker = RenderWorker(self.state) if show else None

        if cap is not None:
            self.add_camera("camera_0", cap)

    def _warmup_models(self):
        warmup_jobs = [
            ("vehicle", self.vehicle_worker.vehicle_model, 640),
            ("plate", self.plate_worker.plate_model, 640),
            ("ocr", self.ocr_worker.char_model, 320),
        ]
        for name, model, imgsz in warmup_jobs:
            warmup = getattr(model, "warmup", None)
            if warmup is None:
                continue
            print(f"[INFO] Warm-up {name} model...")
            t0 = time.perf_counter()
            warmup(imgsz=imgsz)
            print(
                f"[INFO] Warm-up {name} model done in "
                f"{time.perf_counter() - t0:.2f}s"
            )

    def add_camera(
        self,
        camera_id,
        source,
        camera_config=None,
        send_vehicle_events=True,
    ):
        with self.lifecycle_lock:
            if camera_id in self.capture_workers:
                raise ValueError(f"Camera already registered: {camera_id}")

            self.state.register_camera(camera_id)
            self.tracking_worker.register_camera(camera_id)
            if self.event_publisher is not None:
                self.event_publisher.register_camera(
                    camera_id,
                    camera_config=camera_config,
                    enabled=send_vehicle_events,
                )
            capture_worker = CaptureWorker(
                camera_id,
                source,
                self.state,
            )
            self.capture_workers[camera_id] = capture_worker

            if self.started:
                capture_worker.start()

    def start(self):
        with self.lifecycle_lock:
            if self.started:
                return

            self._warmup_models()
            self.started = True
            self.tracking_worker.start()
            self.vehicle_worker.start()
            self.plate_worker.start()
            self.ocr_worker.start()
            if self.render_worker is not None:
                self.render_worker.start()
            for worker in self.capture_workers.values():
                if not worker.is_alive():
                    worker.start()

    def stop_camera(self, camera_id, timeout=2.0):
        with self.lifecycle_lock:
            worker = self.capture_workers.pop(camera_id, None)

        if worker is None:
            return []

        worker.stop()
        if worker.is_alive():
            worker.join(timeout=timeout)

        results = self.get_results(camera_id)
        if self.event_publisher is not None:
            self.event_publisher.send_results(
                camera_id,
                results,
                eligible_only=False,
            )
        self.tracking_worker.unregister_camera(camera_id)
        self.state.unregister_camera(camera_id)
        return results

    def run_until_complete(self, poll_interval=0.1):
        self.start()
        idle_checks = 0
        try:
            while self.state.running:
                if self.state.all_cameras_finished():
                    queues_idle = all(
                        queue.unfinished_tasks == 0
                        for queue in (
                            self.state.vehicle_task_queue,
                            self.state.plate_task_queue,
                            self.state.char_task_queue,
                            self.state.result_queue,
                        )
                    )
                    if (
                        queues_idle
                        and self.state.all_frame_queues_empty()
                        and not self.tracking_worker.has_pending_work()
                    ):
                        idle_checks += 1
                        if idle_checks >= 3:
                            break
                    else:
                        idle_checks = 0
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            self.join()
        return self.get_all_results()

    def stop(self):
        self.state.stop()
        with self.lifecycle_lock:
            workers = list(self.capture_workers.values())
        for worker in workers:
            worker.stop()

    def join(self, timeout=2.0):
        workers = list(self.capture_workers.values())
        workers.extend(
            [
                self.vehicle_worker,
                self.plate_worker,
                self.ocr_worker,
                self.tracking_worker,
            ]
        )
        if self.render_worker is not None:
            workers.append(self.render_worker)

        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=timeout)

        if (
            self.event_publisher is not None
            and not self.event_publisher_closed
        ):
            self.event_publisher.close(self.get_all_results())
            self.event_publisher_closed = True

    def get_latest_frame(self, camera_id):
        frame, _ = self.get_latest_frame_info(camera_id)
        return frame

    def get_latest_frame_id(self, camera_id):
        snapshot = self.state.get_camera(camera_id)
        if snapshot is None:
            return -1
        with snapshot.lock:
            if snapshot.latest_render_frame is not None:
                return snapshot.latest_render_frame_id
            return snapshot.latest_frame_id

    def get_latest_frame_info(self, camera_id):
        snapshot = self.state.get_camera(camera_id)
        if snapshot is None:
            return None, -1
        with snapshot.lock:
            if snapshot.latest_render_frame is not None:
                frame = snapshot.latest_render_frame
                frame_id = snapshot.latest_render_frame_id
            else:
                frame = snapshot.latest_frame
                frame_id = snapshot.latest_frame_id
            return (frame.copy(), frame_id) if frame is not None else (None, -1)

    def get_results(self, camera_id):
        snapshot = self.state.get_camera(camera_id)
        if snapshot is None:
            return []
        with snapshot.lock:
            return list(snapshot.latest_results)

    def get_all_results(self):
        with self.lifecycle_lock:
            camera_ids = list(self.capture_workers.keys())
        return {
            camera_id: self.get_results(camera_id)
            for camera_id in camera_ids
        }

    def get_output(self, timeout=None):
        try:
            return self.state.output_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_error(self, timeout=None):
        try:
            return self.state.error_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_tracking_manager(self, camera_id):
        return self.tracking_worker.get_tracking_manager(camera_id)

    def get_stats(self, camera_id=None):
        return self.state.get_stats(camera_id)
