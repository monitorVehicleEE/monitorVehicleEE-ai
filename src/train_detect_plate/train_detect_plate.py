from ultralytics import YOLO

def main():
    model = YOLO("yolo11s-pose.pt")

    results = model.train(
        data = "./src/train_detect_plate/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,        # 16
        device=0,       # "cpu"
        workers=0,      # 4 số luồng đọc dữ liệu
        project="runs_detect_plate",  # thư mục lưu kết quả
        name="yl11s_dp_ver1"
    )

    print("Training finished. Best weights at:")
    print(results.save_dir / "weights" / "best.pt")  # thư mục lưu model

if __name__ == "__main__":
    main()