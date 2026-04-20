from pathlib import Path

folder = Path("./dataset/vehicle/CAVI-14/CAVI-14/train/")

count = len(list(folder.iterdir()))
print(count)
