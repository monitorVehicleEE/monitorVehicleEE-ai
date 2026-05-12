import cv2
from MainPipeline import MainPipeline, run_on_image, run_on_video
from VehicleDetector import VehicleDetector
from PlateDetector import PlateDetector
from PlateChar import PlateChar

detector_vehicle = VehicleDetector("./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt")
detector_plate = PlateDetector("./runs/pose/runs_detect_plate/yl11s_dp_ver3/weights/best.pt")
detector_char = PlateChar("./runs/detect/runs_read_plate/yolo11s_read_plate_v6/weights/best.pt")

pipeline = MainPipeline(detector_vehicle, detector_plate, detector_char)

#img_out, res = run_on_image("./dataset/vehicle/data_raw/images/train/61/61_00121.jpg", pipeline, save_path="./dataset/output_test", show=True)
# print(res)
video_ot,res = run_on_video("./dataset/vehicle/videos/46.mp4", pipeline, save_path="./dataset/output_test", show=True)
# # print(res)