import os
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


src_root    = Path("./dataset/data_warped/data_raw")
target_root = Path("./dataset/data_warped/data_masked")



def fill_and_remove_boxes(img_el, images_root: Path, output_root: Path,
                          dif_prefix="dif_", target_class_id=32, image_id_filter=None):
    """
    img_el: element <image> trong XML
    images_root: Path tới thư mục chứa ảnh gốc
    output_root: Path tới thư mục lưu ảnh sau khi fill trắng
    dif_prefix: tiền tố attribute dif_
    target_class_id: class id cần fill trắng (32)
    image_id_filter: nếu khác None thì chỉ xử lý image có id này
    
    Returns: list các box_el cần xóa
    """
    img_id = int(img_el.attrib.get("id", -1))

    if (image_id_filter is not None) and (img_id != image_id_filter):
        return []

    img_name = img_el.attrib["name"]
    img_path = images_root / img_name

    if not img_path.is_file():
        print(f"[WARN] Không tìm thấy ảnh: {img_path}")
        return []

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    boxes_to_remove = []

    # Duyệt qua các box
    for box_el in img_el.findall("box"):
        should_fill = False
        
        # Kiểm tra class id = 32
        label = box_el.attrib.get("label", "")
        if label == str(target_class_id):
            should_fill = True
        
        # Kiểm tra dif_* = true
        if not should_fill:
            for attr in box_el.findall("attribute"):
                attr_name = attr.attrib.get("name", "")
                attr_val = (attr.text or "").strip().lower()
                if attr_name.startswith(dif_prefix) and attr_val == "true":
                    should_fill = True
                    break

        if not should_fill:
            continue

        # Lấy toạ độ bbox
        xtl = float(box_el.attrib["xtl"])
        ytl = float(box_el.attrib["ytl"])
        xbr = float(box_el.attrib["xbr"])
        ybr = float(box_el.attrib["ybr"])

        # Fill trắng vùng bbox
        draw.rectangle([xtl, ytl, xbr, ybr], fill=(255, 255, 255))
        
        # Đánh dấu box này để xóa
        boxes_to_remove.append(box_el)

    # Lưu ảnh
    out_path = output_root / img_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    
    if boxes_to_remove:
        print(f"[OK] Image id={img_id}, file={out_path}, đã fill và xóa {len(boxes_to_remove)} boxes")
    
    return boxes_to_remove



def process_cvat_file(xml_path: Path, images_root: Path, output_root: Path,
                      output_xml_path: Path, image_id_filter=None, 
                      dif_prefix="dif_", target_class_id=32):
    """
    xml_path: path file XML export kiểu 'CVAT for images'
    images_root: thư mục chứa ảnh gốc
    output_root: thư mục lưu ảnh output
    output_xml_path: path file XML mới sau khi xóa boxes
    image_id_filter: chỉ xử lý image id cụ thể (None = tất cả)
    dif_prefix: tiền tố attribute dif_
    target_class_id: class id cần xóa (32)
    """
    print(f"=== Đang xử lý XML: {xml_path} ===")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    output_root.mkdir(parents=True, exist_ok=True)
    output_xml_path.parent.mkdir(parents=True, exist_ok=True)

    total_removed = 0

    for img_el in root.findall("image"):
        boxes_to_remove = fill_and_remove_boxes(
            img_el=img_el,
            images_root=images_root,
            output_root=output_root,
            dif_prefix=dif_prefix,
            target_class_id=target_class_id,
            image_id_filter=image_id_filter
        )
        
        # Xóa các boxes khỏi XML
        for box_el in boxes_to_remove:
            img_el.remove(box_el)
            total_removed += 1

    # Lưu XML mới
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Đã xóa tổng cộng {total_removed} boxes khỏi annotation")
    print(f"[INFO] Đã lưu XML mới tại: {output_xml_path}")



if __name__ == "__main__":
    # paths cho data_raw
    images_train = src_root / "images" / "train"
    images_val   = src_root / "images" / "val"
    images_test  = src_root / "images" / "test"

    labels_train = src_root / "labels" / "train" / "annotations_train.xml"
    labels_val   = src_root / "labels" / "val"   / "annotations_val.xml"
    labels_test  = src_root / "labels" / "test"  / "annotations_test.xml"

    # paths cho data_masked (output)
    masked_images_train = target_root / "images" / "train"
    masked_images_val   = target_root / "images" / "val"
    masked_images_test  = target_root / "images" / "test"
    
    masked_labels_train = target_root / "labels" / "train" / "annotations_train.xml"
    masked_labels_val   = target_root / "labels" / "val"   / "annotations_val.xml"
    masked_labels_test  = target_root / "labels" / "test"  / "annotations_test.xml"

    tasks = [
        (labels_train, images_train, masked_images_train, masked_labels_train),
        (labels_val,   images_val,   masked_images_val,   masked_labels_val),
        (labels_test,  images_test,  masked_images_test,  masked_labels_test),
    ]

    for xml_file, img_root, out_img_root, out_xml in tasks:
        process_cvat_file(
            xml_path=xml_file,
            images_root=img_root,
            output_root=out_img_root,
            output_xml_path=out_xml,
            image_id_filter=None,   # =33 nếu bạn muốn chỉ id 33, None nếu tất cả
            dif_prefix="dif_",
            target_class_id=32
        )