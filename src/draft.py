#  
#     def _publish_snapshot(self, camera_id, frame=None):
#         runtime = self.cameras[camera_id]
#         snapshot = self.state.get_camera(camera_id)
#         if snapshot is None:
#             return

#         with snapshot.lock:
#             if snapshot.latest_frame is not None:
#                 base_frame = snapshot.latest_frame.copy()
#                 base_frame_id = snapshot.latest_frame_id
#             elif frame is not None:
#                 base_frame = frame.copy()
#                 base_frame_id = runtime.last_vehicle_result_frame
#             else:
#                 return