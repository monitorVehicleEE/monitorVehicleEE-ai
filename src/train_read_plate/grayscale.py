import cv2
from pathlib import Path

# ROOT dataset
src_root = Path("./dataset/data_warped/data_masked/images")
dst_root = Path("./dataset/data_warped/data_train/images")

# các extension ảnh hợp lệ
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_to_gray(image_path: Path, output_path: Path):
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[SKIP] Không đọc được: {image_path}")
        return

    # convert grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert lại 3 channel để YOLO dùng được
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), gray_3ch)

    print(f"[OK] {output_path}")


def process_folder(root_src: Path, root_dst: Path):
    for path in root_src.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXT:
            rel_path = path.relative_to(root_src)
            out_path = root_dst / rel_path

            convert_to_gray(path, out_path)


if __name__ == "__main__":
    print("=== START CONVERT TO GRAYSCALE ===")
    process_folder(src_root, dst_root)
    print("=== DONE ===")