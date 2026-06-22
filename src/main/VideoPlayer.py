import cv2
import numpy as np


class VideoPlayer:

    def __init__(self, video_path):

        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        # ===== VIDEO INFO =====
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # ===== PLAYER STATE =====
        self.current_frame = 0
        self.paused = False

        # ===== PLAYBACK SPEED =====
        self.speed = 1.0
        self.min_speed = 0.25
        self.max_speed = 8.0

        # ===== DISPLAY SIZE =====
        # chỉnh theo màn hình laptop
        self.max_display_width = 1000
        self.max_display_height = 900

        print("========== VIDEO INFO ==========")
        print(f"FPS           : {self.fps}")
        print(f"Resolution    : {self.width}x{self.height}")
        print(f"Total Frames  : {self.total_frames}")
        print("================================")

        # ===== WINDOW =====
        cv2.namedWindow(
            "Video Player",
            cv2.WINDOW_NORMAL
        )

    # =========================================================
    # RESIZE KEEP ASPECT RATIO
    # =========================================================
    def resize_keep_ratio(self, frame):

        h, w = frame.shape[:2]

        scale_w = self.max_display_width / w
        scale_h = self.max_display_height / h

        scale = min(scale_w, scale_h)

        # không upscale video nhỏ
        scale = min(scale, 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # canvas đen
        canvas = np.zeros(
            (
                self.max_display_height,
                self.max_display_width,
                3
            ),
            dtype=np.uint8
        )

        # center video
        x_offset = (
            self.max_display_width - new_w
        ) // 2

        y_offset = (
            self.max_display_height - new_h
        ) // 2

        canvas[
            y_offset:y_offset + new_h,
            x_offset:x_offset + new_w
        ] = resized

        return canvas

    # =========================================================
    # SEEK FRAME
    # =========================================================
    def seek_frames(self, offset_frames):

        target = self.current_frame + offset_frames

        target = max(
            0,
            min(target, self.total_frames - 1)
        )

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            target
        )

        self.current_frame = target

    # =========================================================
    # SEEK SECOND
    # =========================================================
    def seek_seconds(self, seconds):

        offset_frames = int(seconds * self.fps)

        self.seek_frames(offset_frames)

    # =========================================================
    # CHANGE SPEED
    # =========================================================
    def change_speed(self, factor):

        self.speed *= factor

        self.speed = max(
            self.min_speed,
            min(self.speed, self.max_speed)
        )

        print(
            f"Playback speed: {self.speed:.2f}x"
        )

    # =========================================================
    # DRAW INFO
    # =========================================================
    def draw_info(self, frame):

        time_sec = self.current_frame / self.fps

        info = [
            f"Frame : {self.current_frame}/{self.total_frames}",
            f"Time  : {time_sec:.2f}s",
            f"Speed : {self.speed:.2f}x",
            f"Pause : {self.paused}"
        ]

        y = 40

        for text in info:

            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            y += 40

        return frame

    # =========================================================
    # MAIN LOOP
    # =========================================================
    def run(self):

        while True:

            # ==========================================
            # PLAY MODE
            # ==========================================
            if not self.paused:

                ret, frame = self.cap.read()

                if not ret:
                    print("End of video")
                    break

                self.current_frame = int(
                    self.cap.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                )

            # ==========================================
            # PAUSE MODE
            # ==========================================
            else:

                current_pos = self.current_frame

                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    current_pos
                )

                ret, frame = self.cap.read()

                if not ret:
                    break

                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    current_pos
                )

            # ==========================================
            # DRAW INFO
            # ==========================================
            frame = self.draw_info(frame)

            # ==========================================
            # RESIZE FIT SCREEN
            # ==========================================
            frame = self.resize_keep_ratio(frame)

            # ==========================================
            # SHOW
            # ==========================================
            cv2.imshow("Video Player", frame)

            # ==========================================
            # PLAYBACK DELAY
            # ==========================================
            delay = int(
                1000 / (self.fps * self.speed)
            )

            delay = max(1, delay)

            key = cv2.waitKey(delay) & 0xFF

            # ==========================================
            # QUIT
            # ==========================================
            if key == ord('q'):
                break

            # ==========================================
            # PAUSE / RESUME
            # ==========================================
            elif key == ord(' '):
                self.paused = not self.paused

            # ==========================================
            # SEEK
            # ==========================================
            elif key == ord('a'):
                self.seek_seconds(-1)

            elif key == ord('d'):
                self.seek_seconds(1)

            elif key == ord('w'):
                self.seek_seconds(-5)

            elif key == ord('e'):
                self.seek_seconds(5)

            # ==========================================
            # FRAME BY FRAME
            # ==========================================
            elif key == ord('n'):
                self.seek_frames(1)

            elif key == ord('b'):
                self.seek_frames(-1)

            # ==========================================
            # SPEED
            # ==========================================
            elif key == ord('+') or key == ord('='):
                self.change_speed(2.0)

            elif key == ord('-') or key == ord('_'):
                self.change_speed(0.5)

            elif key == ord('r'):

                self.speed = 1.0

                print(
                    "Playback speed reset to 1.0x"
                )

        self.cap.release()

        cv2.destroyAllWindows()


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":

    video_path = (
        r"./dataset/output_test/27_test.mp4"
    )

    player = VideoPlayer(video_path)

    player.run()