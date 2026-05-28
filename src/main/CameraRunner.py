import cv2
import json
import os
import numpy as np
import time


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
                 drop_late_frames=False, max_frame_skip=3):
        self.cam_id = cam_id
        self.video_source = video_source
        self.pipeline = pipeline
        self.save_dir = save_dir
        self.show = show
        self.drop_late_frames = drop_late_frames
        self.max_frame_skip = max_frame_skip

        self.cap = None
        self.writer = None

        self.frame_idx = 0
        self.running = False

        self.orig_fps = 30.0
        self.frame_interval = 1.0 / self.orig_fps

        self.latest_frame = None
        self.latest_frame_idx = -1

        self.on_finish = None

        self.finalized = False

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
        out_path = self.make_unique_path(
            os.path.join(
                self.save_dir,
                f"{self.cam_id}_tracked.mp4"
            )
        )

        self.writer = cv2.VideoWriter(
            out_path,
            fourcc,
            self.orig_fps,
            (w, h)
        )

        # Đưa lại cap về frame đầu tiên
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.running = True
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

                self.latest_frame = out_frame
                self.latest_frame_idx = self.frame_idx

                # Dọn track cũ theo chu kỳ (nếu cần)
                if self.frame_idx > 0 and self.frame_idx % 30 == 0:
                    self.pipeline.tracking_manager.remove_expired_tracks(
                        self.frame_idx
                    )

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

        json_path = self.make_unique_path(
            os.path.join(
                self.save_dir,
                f"{self.cam_id}_tracked.json"
            )
        )

        print("[INFO] Saving JSON:", json_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable({
                "camera_id": self.cam_id,
                "video_source": self.video_source,
                "total_frames": self.frame_idx,
                "total_vehicles": len(results),
                "vehicles": results,
            }), f, ensure_ascii=False, indent=2)

        self.pipeline.tracking_manager.clear_finalized()

        return results
