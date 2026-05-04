from pathlib import Path

# cần cụ thể folder không là sẽ xóa mất
# labels_path = Path("./dataset/vehicle/data_raw/labels/train/68/")
# images_path = Path("./dataset/vehicle/data_raw/images/train/68/")

# labels_path = Path("./dataset/plate-detect/data_raw/labels/val/")
# images_path = Path("./dataset/plate-detect/data_raw/images/val/")

labels_path = Path("./dataset/data_warped/data_masked/labels/test/")
images_path = Path("./dataset/data_warped/data_masked/images/test/")

deleted_images = 0
deleted_labels = 0

for img_file in images_path.rglob("*"):
    if not img_file.is_file():
        continue

    label_file = labels_path / f"{img_file.stem}.txt"

    # 1) Nếu có label -> kiểm tra rỗng
    if label_file.exists():
        text = label_file.read_text(encoding="utf-8").strip()
        if text == "":
            # label rỗng: xóa label trước, rồi xóa ảnh
            label_file.unlink()
            deleted_labels += 1
            img_file.unlink()
            deleted_images += 1
        # nếu không rỗng -> giữ cả ảnh lẫn label
        continue

    # 2) Nếu không có file label -> xóa ảnh
    img_file.unlink()
    deleted_images += 1

print("Đã xóa ảnh:", deleted_images)
print("Đã xóa label rỗng:", deleted_labels)