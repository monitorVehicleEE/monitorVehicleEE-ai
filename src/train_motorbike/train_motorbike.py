from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data = "motorbike.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,      # số luồng đọc dữ liệu
        project="runs_motorbike",  # thư mục lưu kết quả
        name="yolo11_motorbike"
    )

    print("Training finished. Best weights at:")
    print(results.save_dir / "weights" / "best.pt")  # thư mục lưu model [web:41]

if __name__ == "__main__":
    main()