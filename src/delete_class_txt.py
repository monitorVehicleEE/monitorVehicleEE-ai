from pathlib import Path

labels_path  = Path("./dataset/vehicle/CAVI-14/CAVI-14/train/")

deleted_count = 0
kept_count = 0

# Duyệt qua tất cả các file .txt trong thư mục và thư mục co
for txt_file  in labels_path.rglob("*.txt"):
    has_class_0  = False
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
        
            class_id = int(float(parts[0]))
            # Nếu tồn tại class khác 0 thì đánh dấu xóa
            if class_id == 0:
                has_class_0  = True
                break

    if not has_class_0:
        txt_file.unlink()
        deleted_count += 1
        print(f"Đã xóa: {txt_file}")
    else:
        kept_count += 1

print("\nHoàn tất!")
print(f"Đã giữ lại: {kept_count} file")
print(f"Đã xóa: {deleted_count} file")