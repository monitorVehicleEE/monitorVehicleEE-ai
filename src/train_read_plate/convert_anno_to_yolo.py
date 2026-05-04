import os
from pathlib import Path
import xml.etree.ElementTree as ET

# gốc cho data_masked (ảnh đã fill trắng)
masked_root = Path("./dataset/data_warped/data_masked")

# mapping label -> class_id
LABEL2ID = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "K": 18,
    "L": 19,
    "M": 20,
    "N": 21,
    "P": 22,
    "S": 23,
    "T": 24,
    "U": 25,
    "V": 26,
    "X": 27,
    "Y": 28,
    "Z": 29,
    "-": 30,
    ".": 31,
    "delete": 32,
    "R": 33,
}

# class_id cần bỏ (delete = 32)
IGNORE_CLASS_IDS = {32}


def box_has_dif_true(box_el, dif_prefix="dif_"):
    for attr in box_el.findall("attribute"):
        attr_name = attr.attrib.get("name", "")
        attr_val = (attr.text or "").strip().lower()
        if attr_name.startswith(dif_prefix) and attr_val == "true":
            return True
    return False


def convert_cvat_to_yolo(xml_path: Path, images_root: Path, labels_root: Path,
                         skip_dif_true=True, dif_prefix="dif_"):
    """
    xml_path: file XML 'CVAT for images'
    images_root: thư mục ảnh (data_masked/images/train|val|test)
    labels_root: thư mục labels output (data_masked/labels/train|val|test)
    skip_dif_true: nếu True thì bỏ mọi box có dif_* = true
    """
    print(f"=== Convert {xml_path} ===")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels_root.mkdir(parents=True, exist_ok=True)

    for img_el in root.findall("image"):
        img_id = int(img_el.attrib.get("id", -1))
        img_name = img_el.attrib["name"]

        img_path = images_root / img_name
        if not img_path.is_file():
            print(f"[WARN] Không tìm thấy ảnh (bỏ qua): {img_path}")
            continue

        img_w = float(img_el.attrib.get("width", 0))
        img_h = float(img_el.attrib.get("height", 0))
        if img_w <= 0 or img_h <= 0:
            from PIL import Image
            with Image.open(img_path) as im:
                img_w, img_h = im.size

        yolo_lines = []

        for box_el in img_el.findall("box"):
            # 1) bỏ box có dif_* = true nếu được yêu cầu
            if skip_dif_true and box_has_dif_true(box_el, dif_prefix=dif_prefix):
                continue

            # 2) lấy label và map sang class_id
            label = box_el.attrib.get("label")
            if label not in LABEL2ID:
                print(f"[WARN] Label {label} không có trong LABEL2ID, bỏ qua.")
                continue
            class_id = LABEL2ID[label]

            # 3) bỏ class delete (class_id = 32)
            if class_id in IGNORE_CLASS_IDS:
                continue

            xtl = float(box_el.attrib["xtl"])
            ytl = float(box_el.attrib["ytl"])
            xbr = float(box_el.attrib["xbr"])
            ybr = float(box_el.attrib["ybr"])

            # chuyển sang YOLO: cx, cy, w, h (relative)
            cx = (xtl + xbr) / 2.0 / img_w
            cy = (ytl + ybr) / 2.0 / img_h
            bw = (xbr - xtl) / img_w
            bh = (ybr - ytl) / img_h

            # clip cho an toàn
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = labels_root / (Path(img_name).stem + ".txt")
        # ghi cả khi không có box (file rỗng) để YOLO hiểu là ảnh không có object
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        print(f"[OK] {label_path} ({len(yolo_lines)} boxes)")


if __name__ == "__main__":
    # thư mục ảnh đã mask
    images_train = masked_root / "images" / "train"
    images_val   = masked_root / "images" / "val"
    images_test  = masked_root / "images" / "test"

    # thư mục labels output tương ứng
    labels_train_out = masked_root / "labels" / "train"
    labels_val_out   = masked_root / "labels" / "val"
    labels_test_out  = masked_root / "labels" / "test"

    # file XML gốc
    src_root = Path("./dataset/data_warped/data_raw")
    xml_train = src_root / "labels" / "train" / "annotations_train.xml"
    xml_val   = src_root / "labels" / "val"   / "annotations_val.xml"
    xml_test  = src_root / "labels" / "test"  / "annotations_test.xml"

    tasks = [
        (xml_train, images_train, labels_train_out),
        (xml_val,   images_val,   labels_val_out),
        (xml_test,  images_test,  labels_test_out),
    ]

    for xml_file, img_root, lbl_root in tasks:
        convert_cvat_to_yolo(
            xml_path=xml_file,
            images_root=img_root,
            labels_root=lbl_root,
            skip_dif_true=True,    # bỏ dif_* = true
            dif_prefix="dif_"
        )