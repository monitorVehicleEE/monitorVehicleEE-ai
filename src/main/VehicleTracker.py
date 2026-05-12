# from ultralytics import YOLO

# class VehicleTracker:
#     def __init__(self, model_path, video_source, tracker_cfg = "bytetrack.yaml"):
#         self.model = YOLO(model_path)
#         self.results_stream = self.model.track(
#             source  = video_source,
#             tracker = tracker_cfg,
#             persist = True,
#             stream  =  True
#         )

#     def next_frame(self):
#         result = next(self.results_stream, None)
#         if result is None:
#             return None, []

#         frame = result.orig_img
#         vehicels