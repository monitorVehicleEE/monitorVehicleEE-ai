from ultralytics import YOLO
import torch

def main():
    # Load mô hình đã train trước đó
    model = YOLO("./runs/detect/runs_motorbike/yolo11_motorbike_gpu_ver3/weights/best.pt")

    # Fine-tune với dữ liệu mới
    results = model.train(
        data="./src/train_motorbike/motorbike.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,  # GPU
        lr0=0.001,
        project="./runs/detect/runs_motorbike/", #runs
        name="finetune_motorbike", 
        exist_ok=True,         # Cho phép ghi đè
        pretrained=True,
        resume=False,
        workers=0  # giảm số worker để ổn định trên Windows
    )

    return results


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # cần thiết trên Windows
    main()
