import os
from pathlib import Path
from PIL import Image, ImageDraw

src_root    = Path("dataset/plate-detect/data_raw")
target_root = Path("dataset/plate-detect/data_masked")

for split in ["train", "val", "test"]:
    src_images_root = src_root / "images" / split
    src_labels_root = src_root / "labels" / split

    image_paths = list(src_images_root.rglob("*.jpg")) + \
                  list(src_images_root.rglob("*.jpeg"))

    for image_path in image_paths:
        relative_image_path = image_path.relative_to(src_images_root)

        target_image_path = target_root / "images" / split / relative_image_path
        target_image_path.parent.mkdir(parents=True, exist_ok=True)

        src_labels_path = src_labels_root / relative_image_path.with_suffix(".txt")
        target_label_path = target_root / "labels" / split / relative_image_path.with_suffix(".txt")
        target_label_path.parent.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        drawer = ImageDraw.Draw(image)

        lines_to_keep = []

        if src_labels_path.exists():
            with open(src_labels_path, "r") as f:
                for line in f.read().strip().splitlines():
                    if not line:
                        continue

                    parts = list(map(float, line.split()))
                    class_id = int(parts[0])

                    # ---- BBOX ----
                    x_center, y_center, box_w, box_h = parts[1:5]

                    x_center *= width
                    y_center *= height
                    box_w *= width
                    box_h *= height

                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    # ---- KEYPOINTS (tự động detect số lượng) ----
                    kpts = parts[5:]

                    if len(kpts) % 3 != 0:
                        print(f"[WARN] Sai format keypoints: {image_path}")
                        continue

                    num_kpts = len(kpts) // 3

                    keypoints = []
                    for i in range(num_kpts):
                        kx = kpts[i*3] * width
                        ky = kpts[i*3+1] * height
                        v  = kpts[i*3+2]

                        keypoints.append((kx, ky, v))

                    # ---- LOGIC ----
                    if class_id == 0:
                        # giữ lại label
                        lines_to_keep.append(line)

                    else:
                        # mask theo bbox
                        drawer.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

                        # ---- OPTIONAL: mask theo polygon (4 keypoints) ----
                        # pts = [(int(kx), int(ky)) for kx, ky, v in keypoints if v > 0]
                        # if len(pts) >= 3:
                        #     drawer.polygon(pts, fill=(255,255,255))

        # lưu ảnh
        image.save(target_image_path)

        # lưu label
        if lines_to_keep:
            with open(target_label_path, "w") as f:
                f.write("\n".join(lines_to_keep))
        else:
            # open(target_label_path, "w").close()
            if target_label_path.exists():
                target_label_path.unlink()
