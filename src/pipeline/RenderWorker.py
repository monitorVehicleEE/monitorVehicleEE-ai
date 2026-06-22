import threading

import cv2


class RenderWorker(threading.Thread):
    """Optional preview worker. Production callers should use result callbacks."""

    def __init__(self, state, refresh_interval=0.02, max_window_size=(1280, 720)):
        super().__init__(name="preview-render", daemon=True)
        self.state = state
        self.refresh_interval = refresh_interval
        self.max_window_size = max_window_size
        self.windows = set()

    def _fit_to_window(self, frame):
        max_width, max_height = self.max_window_size
        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 1.0:
            return frame

        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        return cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    def run(self):
        while self.state.running:
            displayed = False
            with self.state.camera_lock:
                camera_items = list(self.state.cameras.items())

            for camera_id, snapshot in camera_items:
                with snapshot.lock:
                    frame = (
                        snapshot.latest_render_frame.copy()
                        if snapshot.latest_render_frame is not None
                        else None
                    )
                if frame is None:
                    continue
                displayed = True
                window_name = f"Vehicle Tracking - {camera_id}"
                display_frame = self._fit_to_window(frame)
                if window_name not in self.windows:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(
                        window_name,
                        display_frame.shape[1],
                        display_frame.shape[0],
                    )
                    self.windows.add(window_name)
                cv2.imshow(window_name, display_frame)

            if displayed and cv2.waitKey(1) & 0xFF == 27:
                self.state.stop()
                break

            self.state.stop_event.wait(self.refresh_interval)

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
