from ultralytics import YOLO

def main():
    model = YOLO("./runs/detect/runs_motorbike/yolo11_motorbike_gpu_ver3/weights/best.pt")

    metrics = model.val(
        data="./src/train_motorbike/motorbike.yaml",
        split="test"
    )

    print(metrics)

if __name__ == "__main__":
    main()
