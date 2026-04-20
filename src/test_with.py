from ultralytics import YOLO

model = YOLO("./runs/pose/runs_detect_plate/yl11s_dp_ver1/weights/best.pt")

results = model.predict( 
    source="./dataset/vehicle/test_videos/VID_20260415_084407.mp4",
    save=True,
    conf=0.25,
    # trường hợp không muốn sinh folder mới
    project="./runs/pose",     # thư mục gốc
    name="video_results",      # tên cố định
    exist_ok=True              # cho phép ghi đè
)
