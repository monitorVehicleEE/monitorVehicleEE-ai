import os
import json
import requests
from tqdm import tqdm

JSON_FILE  = "./dataset/vehicle/Motocycle/images_urls_1.json"
SAVE_DIR  = "./dataset/vehicle/Motocycle/images/train/"
BATCH_SIZE = 100
START_INDEX = 0 # vị trí tải
END_INDEX = 500 # tải đến đâu

headers = {"User-Agent": "Mozilla/5.0"} # giả lập req trình duyệt

# LOAD JSON
with open(JSON_FILE,'r', encoding='utf-8') as f:
    urls = json.load(f)

# DOWNLOAD
for i in tqdm(range(START_INDEX, min(END_INDEX, len(urls)))):
    url = urls[i]

    filename = url.split("plain/")[-1]

    # xác định folder
    batch_id = i // BATCH_SIZE
    batch_folder =  os.path.join(SAVE_DIR,f"batch_{batch_id}")

    os.makedirs(batch_folder, exist_ok=True)

    save_path = os.path.join(batch_folder, filename)

    # đã tải thì bỏ qua
    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(url, headers = headers, timeout = 10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f: # đường dẫn - write binary vì ảnh là file nhị phân
                f.write(response.content)
    except Exception as e:
        print("Lỗi: ", url)
