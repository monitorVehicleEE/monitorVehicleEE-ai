import cv2
from MainPipeline import MainPipeline, run_on_image
from CameraRunner import CameraRunner
from VehicleTracker import VehicleTracker
from PlateDetector import PlateDetector
from PlateChar import PlateChar
from VehicleDetector import VehicleDetector

# 1. Khởi tạo model
video_path = "./dataset/vehicle/videos/27.mp4"

detector_vehicle = VehicleDetector(
    "./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt"
)
tracker_vehicle  = VehicleTracker()
detector_plate   = PlateDetector(
    "./runs/pose/runs_detect_plate/yl11s_dp_ver6/weights/best.pt"
)
detector_char    = PlateChar(
    "./runs/detect/runs_read_plate/yolo11s_read_plate_v6/weights/best.pt"
)

# 2. Tạo pipeline
pipeline = MainPipeline(
    vehicle_detector=detector_vehicle,
    vehicle_tracker=tracker_vehicle,
    plate_detector=detector_plate,
    char_detector=detector_char
)

# img_out, res = run_on_image("./dataset/vehicle/frames/61/61_00121.jpg", pipeline, save_path="./dataset/output_test", show=True)

# 3. Tạo CameraRunner cho 1 cam

runner = CameraRunner(
    cam_id="cam_27",
    video_source=video_path,
    pipeline=pipeline,
    save_dir="./dataset/output_test",
    show=True
)

# 4. Chạy
runner.setup()
runner.run_loop()
results = runner.finalize()

print("Tổng số vehicle:", len(results))