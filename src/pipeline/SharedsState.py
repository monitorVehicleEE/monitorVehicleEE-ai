import threading
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Dict, Optional

import numpy as np

@dataclass
class CameraSnapshot:
    frame_queue: Queue = field(default_factory=lambda: Queue(maxsize=2))
    latest_frame: Optional[np.ndarray] = None
    latest_render_frame: Optional[np.ndarray] = None
    latest_frame_id: int = -1
    latest_render_frame_id: int = -1
    latest_results: list = field(default_factory=list)
    finished: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


class SharedState:
    def __init__(
        self,
        frame_queue_size=2,
        vehicle_queue_size=4,
        anpr_queue_size=16,
        char_queue_size=None,
        result_queue_size=64,
        output_queue_size=16,
    ):
        self.frame_queue_size = frame_queue_size
        self.vehicle_task_queue = Queue(maxsize=vehicle_queue_size)
        self.plate_task_queue = Queue(maxsize=anpr_queue_size)
        self.char_task_queue = Queue(maxsize=char_queue_size or anpr_queue_size)
        self.result_queue = Queue(maxsize=result_queue_size)
        self.output_queue = Queue(maxsize=output_queue_size)
        self.error_queue = Queue()

        self.stop_event = threading.Event()
        self.cameras: Dict[str, CameraSnapshot] = {}
        self.camera_lock = threading.RLock()
        self.stats = {}
        self.stats_lock = threading.RLock()

    @property
    def running(self):
        return not self.stop_event.is_set()

    def stop(self):
        self.stop_event.set()

    def report_error(self, worker_name, camera_id, error):
        self.error_queue.put((worker_name, camera_id, error))

    def register_camera(self, camera_id):
        with self.camera_lock:
            if camera_id in self.cameras:
                raise ValueError(f"Camera already registered: {camera_id}")
            self.cameras[camera_id] = CameraSnapshot(
                frame_queue=Queue(maxsize=self.frame_queue_size)
            )
        with self.stats_lock:
            self.stats.setdefault(
                str(camera_id),
                {
                    "captured_frames": 0,
                    "frame_queue_dropped": 0,
                    "vehicle_tasks": 0,
                    "vehicle_task_dropped": 0,
                    "vehicle_results": 0,
                    "plate_tasks": 0,
                    "plate_task_dropped": 0,
                    "plate_results": 0,
                    "char_tasks": 0,
                    "char_task_dropped": 0,
                    "char_results": 0,
                    "renders": 0,
                },
            )

    def unregister_camera(self, camera_id):
        with self.camera_lock:
            snapshot = self.cameras.pop(camera_id, None)
        if snapshot is None:
            return

        while True:
            try:
                snapshot.frame_queue.get_nowait()
                snapshot.frame_queue.task_done()
            except Empty:
                break
        with self.stats_lock:
            self.stats.pop(str(camera_id), None)

    def get_camera(self, camera_id):
        with self.camera_lock:
            return self.cameras.get(camera_id)

    def all_cameras_finished(self):
        with self.camera_lock:
            return bool(self.cameras) and all(
                snapshot.finished for snapshot in self.cameras.values()
            )

    def all_frame_queues_empty(self):
        with self.camera_lock:
            return all(
                snapshot.frame_queue.empty()
                for snapshot in self.cameras.values()
            )

    def record_stat(self, camera_id, key, delta=1):
        with self.stats_lock:
            camera_stats = self.stats.setdefault(str(camera_id), {})
            camera_stats[key] = camera_stats.get(key, 0) + delta

    def get_stats(self, camera_id=None):
        with self.stats_lock:
            if camera_id is None:
                return {
                    key: dict(value)
                    for key, value in self.stats.items()
                }
            return dict(self.stats.get(str(camera_id), {}))


#dùng để đưa item mới vào queue mà không chờ
# và nếu queue đang đầy thì bỏ item cũ nhất để nhường chỗ cho item mới.
def put_latest(queue: Queue, item) -> bool:
    try:
        queue.put_nowait(item)
        return True
    except Full:
        try:
            queue.get_nowait()
            queue.task_done()
        except Empty:
            return False

    try:
        queue.put_nowait(item)
        return True
    except Full:
        return False


def put_if_available(queue: Queue, item) -> bool:
    # Put without blocking; return False when the queue is saturated.
    try:
        queue.put_nowait(item)
        return True
    except Full:
        return False


def put_until_stopped(queue: Queue, item, stop_event, timeout=0.2) -> bool:
    while not stop_event.is_set():
        try:
            queue.put(item, timeout=timeout)
            return True
        except Full:
            continue
    return False
