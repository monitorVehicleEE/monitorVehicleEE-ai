from pathlib import Path
import shutil

# Thư mục nguồn và đích
source_dir = Path("./dataset/vehicle/CAVI-14/CAVI-14/test/")  
dest_dir = Path("./dataset/vehicle/oto_raw/labels/test/")       # thư mục đích

# Tạo thư mục đích nếu chưa có
dest_dir.mkdir(parents=True, exist_ok=True)

copied_count = 0

# Duyệt tất cả file .txt
for txt_file in source_dir.rglob("*.txt"):
    dest_file = dest_dir / txt_file.name  # giữ nguyên tên file

    shutil.copy2(txt_file, dest_file)
    copied_count += 1
    print(f"Đã copy: {txt_file} -> {dest_file}")

print("\nHoàn tất!")
print(f"Tổng file đã copy: {copied_count}")
