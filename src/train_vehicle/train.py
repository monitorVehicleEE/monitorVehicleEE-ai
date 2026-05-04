from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    results = model.train(
        data = "./src/train_vehicle/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,        # 16
        device=0,       # "cpu"
        workers=0,      # 4 số luồng đọc dữ liệu
        project="runs_vehicle",  # thư mục lưu kết quả
        name="yolo11s_vehicle_v1"
    )

    print("Training finished. Best weights at:")
    print(results.save_dir / "weights" / "best.pt")  # thư mục lưu model

if __name__ == "__main__":
    main()