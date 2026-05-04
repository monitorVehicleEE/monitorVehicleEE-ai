import json
from pathlib import Path
from PIL import Image, ImageDraw

# ====== CONFIG ======
coco_path   = Path("./src/train_read_plate/instances_Train.json")

src_root    = Path("./dataset/data_warped/data_raw")
target_root = Path("./dataset/data_warped/data_masked")

splits = ["train", "val", "test"]

# ====== LOAD COCO ======
with open(coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# ====== MAP: file_name -> image_id (CHỈ LẤY TÊN FILE) ======
file_to_id = {}
for img in coco["images"]:
    name = img["file_name"].replace("\\", "/").split("/")[-1]
    file_to_id[name] = img["id"]

# ====== GROUP ANNOTATIONS ======
anns_per_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    anns_per_image.setdefault(img_id, []).append(ann)

# ====== MAP CATEGORY ======
cat_id_map = {}
for i, cat in enumerate(coco["categories"]):
    cat_id_map[cat["id"]] = i

print(f"Total images in COCO: {len(file_to_id)}")

# ====== PROCESS ======
for split in splits:
    print(f"\nProcessing split: {split}")

    src_images_root = src_root / "images" / split
    target_img_root = target_root / "images" / split
    target_lbl_root = target_root / "labels" / split

    image_paths = list(src_images_root.rglob("*.jpg")) + \
                  list(src_images_root.rglob("*.jpeg"))

    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        relative_path = image_path.relative_to(src_images_root)

        target_image_path = target_img_root / relative_path
        target_label_path = target_lbl_root / relative_path.with_suffix(".txt")

        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        target_label_path.parent.mkdir(parents=True, exist_ok=True)

        # ====== LOAD IMAGE ======
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)

        # ====== FIND IMAGE_ID ======
        image_id = file_to_id.get(image_path.name)

        if image_id is None:
            print(f"NOT FOUND: {image_path.name}")
            image.save(target_image_path)
            open(target_label_path, "w").close()
            continue

        anns = anns_per_image.get(image_id, [])
        lines_to_keep = []

        # ====== PROCESS ANNOTATIONS ======
        for ann in anns:
            x, y, bw, bh = ann["bbox"]

            # clamp bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(width, int(x + bw))
            y2 = min(height, int(y + bh))

            # ====== READ ATTRIBUTE DIF ======
            is_dif = False
            attrs = ann.get("attributes", {})

            if isinstance(attrs, dict):
                is_dif = attrs.get("dif", False)
            elif isinstance(attrs, list):
                for attr in attrs:
                    if attr.get("name") == "dif" and attr.get("value") == True:
                        is_dif = True
                        break

            # ====== HANDLE DIF ======
            if is_dif or ann["category_id"] == 33:
                # fill trắng
                draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
            else:
                # convert YOLO
                cx = (x + bw / 2) / width
                cy = (y + bh / 2) / height
                nw = bw / width
                nh = bh / height

                class_id = cat_id_map[ann["category_id"]]

                lines_to_keep.append(f"{class_id} {cx} {cy} {nw} {nh}")

        # ====== SAVE ======
        image.save(target_image_path)

        if lines_to_keep:
            with open(target_label_path, "w") as f:
                f.write("\n".join(lines_to_keep))
        else:
            open(target_label_path, "w").close()

print("\n✅ DONE!")
