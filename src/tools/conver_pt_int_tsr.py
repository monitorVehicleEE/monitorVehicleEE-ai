from ultralytics import YOLO

model = YOLO("./model/pytorch/vehicle/best.pt")
model.export(format="onnx", device=0, opset=12)