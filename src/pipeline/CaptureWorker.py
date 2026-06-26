import threading
import time

import cv2

from src.pipeline.FrameData import FrameData
from src.pipeline.SharedsState import put_latest


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        camera_id,
        source,
        state,
        reconnect_delay=1.0,
        pace_file=True,
    ):
        super().__init__(name=f"capture-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.state = state
        self.reconnect_delay = reconnect_delay
        self.pace_file = pace_file
        self.cap = source if hasattr(source, "read") else None
        self.owns_capture = self.cap is None
        self.local_stop_event = threading.Event()

    def stop(self):
        self.local_stop_event.set()

    def _open(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.source)
        return self.cap is not None and self.cap.isOpened()

    def _close(self):
        if self.cap is not None and self.owns_capture:
            self.cap.release()
        self.cap = None

    def run(self):
        frame_id = 0
        snapshot = self.state.get_camera(self.camera_id)
        next_frame_at = time.perf_counter()
        
        # 1. FPS mục tiêu 
        
        # Biến tracking thời gian để quyết định khi nào giữ hoặc bỏ frame
        target_fps = 15.0
        # target_interval = 1.0 / target_fps
        last_processed_time = 0.0 

        try:
            if not self._open():
                raise ValueError(f"Cannot open source for camera {self.camera_id}")

            source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
            target_interval = 1.0 / target_fps
            
            # frame_interval này dùng để điều tốc khi đọc FILE tránh đọc quá nhanh
            frame_interval = (
                1.0 / source_fps
                if self.pace_file and self._is_file_source() and source_fps > 0
                else 0.0
            )

            while self.state.running and not self.local_stop_event.is_set():
                if frame_interval > 0:
                    delay = next_frame_at - time.perf_counter()
                    if delay > 0 and self._wait(delay):
                        break

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if isinstance(self.source, (str, bytes)) and not self._is_file_source():
                        self._close()
                        if self._wait(self.reconnect_delay):
                            break
                        if not self._open():
                            continue
                        continue
                    break
                # if frame_id == 0:
                #     print("[CAPTURE BEFORE]", frame.shape)
                # frame = self.resize_frame(frame, 640)
                # if frame_id == 0:
                #     print("[CAPTURE AFTER]", frame.shape)
                # LỌC KHUNG HÌNH THEO TARGET FPS 
                current_time = time.time()
                # Nếu khoảng cách từ frame đã xử lý trước đó chưa đủ thời gian của target_fps, BỎ QUA frame này
                if current_time - last_processed_time < target_interval:
                    if frame_interval > 0: # Cập nhật thời gian cho vòng lặp file
                        next_frame_at = max(next_frame_at + frame_interval, time.perf_counter())
                    continue 
                
                # Cập nhật mốc thời gian của frame được giữ lại
                last_processed_time = current_time
                # ---------------------------------------------------

                packet = FrameData(
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    timestamp=current_time, # Sử dụng luôn biến current_time tối ưu hơn
                    frame=frame,
                )
                if snapshot is not None:
                    with snapshot.lock:
                        snapshot.latest_frame = frame.copy()
                        snapshot.latest_frame_id = frame_id
                    if not put_latest(snapshot.frame_queue, packet):
                        self.state.record_stat(self.camera_id, "frame_queue_dropped")
                self.state.record_stat(self.camera_id, "captured_frames")

                frame_id += 1
                if frame_interval > 0:
                    next_frame_at = max(
                        next_frame_at + frame_interval,
                        time.perf_counter(),
                    )
        except Exception as exc:
            self.state.report_error(self.name, self.camera_id, exc)
        finally:
            if snapshot is not None:
                with snapshot.lock:
                    snapshot.finished = True
            self._close()

    def _is_file_source(self):
        if not isinstance(self.source, str):
            return False
        lower = self.source.lower()
        return lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

    def _wait(self, timeout):
        deadline = time.perf_counter() + timeout
        while timeout > 0:
            if (
                self.local_stop_event.is_set()
                or self.state.stop_event.wait(min(timeout, 0.1))
            ):
                return True
            timeout = deadline - time.perf_counter()
        return (
            self.local_stop_event.is_set()
            or self.state.stop_event.is_set()
        ) 
    
    @staticmethod
    def resize_frame(frame, max_side=640):
        h, w = frame.shape[:2]

        longest = max(h, w)

        if longest <= max_side:
            return frame

        scale = max_side / longest

        return cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )