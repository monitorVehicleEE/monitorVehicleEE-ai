from ultralytics import YOLO

def main():
    model = YOLO("./runs/detect/runs_read_plate/yolo11s_read_plate_v5/weights/best.pt")

    metrics = model.val(
        data="./src/train_read_plate/data.yaml",
        split="test",
        imgsz=640,
        batch=4,
        device=0
    )

    print(metrics)

if __name__ == "__main__":
    main()
