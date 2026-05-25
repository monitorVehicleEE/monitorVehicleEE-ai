import cv2
import json
import os


class CameraRunner:
    def __init__( self, cam_id, video_source, pipeline, save_dir="./output", show=False ):
        self.cam_id = cam_id
        self.video_source = video_source
        self.pipeline = pipeline
        self.save_dir = save_dir
        self.show = show

        self.cap = None
        self.writer = None

        self.frame_idx = 0
        self.running = False

        self.orig_fps = 30.0

        self.latest_frame = None

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

            raise ValueError(
                f"Cannot open source: {self.video_source}"
            )

        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        self.running = True

        self.frame_idx = 0

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run_loop(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[INFO] Video ended: {self.cam_id}")
                    break

                self.pipeline.frame_index = self.frame_idx

                out_frame, results = (
                    self.pipeline.process_frame_with_tracked_vehicles(frame)
                )

                self.latest_frame = out_frame

                if (
                    self.frame_idx > 0
                    and self.frame_idx % 30 == 0
                ):
                    self.pipeline.tracking_manager.remove_expired_tracks(
                        self.frame_idx
                    )

                if self.writer is not None:
                    try:
                        self.writer.write(out_frame)
                    except Exception as e:
                        print("[WRITER ERROR]", e)

                if self.show:

                    cv2.imshow(
                        f"Cam {self.cam_id}",
                        out_frame
                    )

                    if cv2.waitKey(1) & 0xFF == 27:

                        self.stop()

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

        results = (
            self.pipeline.tracking_manager.export_all_results()
        )

        json_path = self.make_unique_path(
            os.path.join(
                self.save_dir,
                f"{self.cam_id}_tracked.json"
            )
        )

        print("[INFO] Saving JSON:", json_path)

        with open(json_path, "w", encoding="utf-8") as f:

            json.dump({
                "camera_id": self.cam_id,
                "video_source": self.video_source,
                "total_frames": self.frame_idx,
                "total_vehicles": len(results),
                "vehicles": results,
            }, f, ensure_ascii=False, indent=2)

        self.pipeline.tracking_manager.clear_finalized()

        return results