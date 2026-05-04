from ultralytics import YOLO

def main():
    # load pretrained pose model
    model = YOLO("yolo11s-pose.pt")

    results = model.train(
        data="./src/train_detect_plate/data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers = 0,
        task="pose",
        # augmentation 
        degrees=5,
        translate=0.05,
        scale=0.5,
        fliplr=0.5,

        project="runs_detect_plate",  # thư mục lưu kết quả
        name="yl11s_dp_ver3"
    )

if __name__ == "__main__":
    main()
