from pathlib import Path

labels_path = Path("./dataset/plate-detect/data_raw/labels/test/")
images_path = Path("./dataset/plate-detect/data_raw/images/test/")

deleted = 0

for img_file in images_path.rglob("*"):
    if not img_file.is_file():
        continue

    label_file = labels_path / f"{img_file.stem}.txt"

    if not label_file.exists():
        img_file.unlink()
        deleted += 1

print("Đã xóa:", deleted)