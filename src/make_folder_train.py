from pathlib import Path

# Thư mục chứa ảnh
image_dir = Path("./dataset/vehicle/oto_raw/images/train/")  

# File output
output_file = Path("./Train.txt")

# Tiền tố đường dẫn ghi vào file
prefix = "data/images/Train/"

# Lấy danh sách ảnh .jpg
image_files = sorted(image_dir.glob("*.jpg"))

# Ghi vào file Train.txt
with open(output_file, "w", encoding="utf-8") as f:
    for img in image_files:
        f.write(prefix + img.name + "\n")

print("Hoàn tất!")
print(f"Đã ghi {len(image_files)} ảnh vào {output_file}")
