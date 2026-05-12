import cv2
from MainPipeline import MainPipeline, run_on_image, run_on_video, run_on_video_with_tracker
from VehicleTracker import VehicleTracker
from PlateDetector import PlateDetector
from PlateChar import PlateChar

video_path = "./dataset/vehicle/test_videos/64.mp4"
vehicle_tracker  = VehicleTracker("./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt", 
                                    video_source=video_path)
detector_plate = PlateDetector("./runs/pose/runs_detect_plate/yl11s_dp_ver3/weights/best.pt")
detector_char = PlateChar("./runs/detect/runs_read_plate/yolo11s_read_plate_v6/weights/best.pt")

# pipeline = MainPipeline(detector_vehicle, detector_plate, detector_char)
pipeline = MainPipeline(vehicle_detector=None, plate_detector=detector_plate, char_detector= detector_char)

#img_out, res = run_on_image("./dataset/vehicle/data_raw/images/train/61/61_00121.jpg", pipeline, save_path="./dataset/output_test", show=True)
# print(res)
#video_ot,res = run_on_video("./dataset/vehicle/videos/46.mp4", pipeline, save_path="./dataset/output_test", show=True)
# # print(res)

# Chạy video với tracker và lưu JSON
all_results = run_on_video_with_tracker(
    video_source=video_path,
    pipeline=pipeline,
    vehicle_tracker=vehicle_tracker,
    save_video_path="./dataset/output_test/46_out_track.mp4",
    save_json_path="./dataset/output_test/46_out_track.json",
    show=True
)

print("Số frame có dữ liệu:", len(all_results))