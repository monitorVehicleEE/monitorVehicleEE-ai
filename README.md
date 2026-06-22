# cài đặt môi trường: python -m venv .venv
<!-- .venv\Scripts\python.exe -m -->
<!-- Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -->
<!-- .\.venv\Scripts\Activate.ps1 -->
<!-- python src/train_motorbike/train_motorbike.py -->
<!-- .\.venv_trt\Scripts\Activate.ps1 -->
# cài package yolo: pip install ultralytics
# cài đặt openCV đọc ảnh/video: pip install opencv-python
# tải ảnh từ url: pip install requests
# hiện thanh bar tiến trình: pip install tqdm
# cài đặt cvat: git clone https://github.com/opencv/cvat
# pillow : pip install pillow
# pip install numpy==1.26.4



### dành cho xài tensorrt ###
# gỡ torch cũ: pip uninstall torch torchvision torchaudio -y
# cài torch phù hợp để export sang tensortRT: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Cài ONNX + TensorRT dependencies: pip install onnx onnxruntime-gpu
# càu tensortRT: pip install tensorrt || pip install tensorrt==10.11.0.33
# trtexec --onnx=model/pytorch/vehicle/best.onnx --saveEngine=best.engine --fp16

# run sequential server
uvicorn src.app.app:app --reload --host 0.0.0.0 --port 8001

# run thread server
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8001

python src/pipeline/run_video.py .\dataset\vehicle\videos\27.mp4 --show --save-video --save-event-images

python src/pipeline/run_video.py .\dataset\vehicle\videos\27.mp4 --show --save-event-images
