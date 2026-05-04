from ultralytics import YOLO

model = YOLO("./runs/detect/runs_vehicle/yolo11s_vehicle_v1/weights/best.pt")

results = model.predict( 
    source="./dataset/vehicle/test_videos/VID_20260426_101157.mp4",
    save=True,
    conf=0.45,
    
    project="./runs/vehicle",     # thư mục gốc
    name="video_results",      # tên cố định
    exist_ok=True              # cho phép ghi đè
)
