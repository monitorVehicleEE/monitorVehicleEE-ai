import cv2
from VehicleDetector import VehicleDetector
from PlateDetector import PlateDetector
from PlateWarper import PlateWarper
from PlateChar import PlateChar

detector_vehicle = VehicleDetector("./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt")
detector_plate = PlateDetector("./runs/pose/runs_detect_plate/yl11s_dp_ver3/weights/best.pt")
detector_char = PlateChar("./runs/detect/runs_read_plate/yolo11s_read_plate_v5/weights/best.pt")

image_path = "./dataset/vehicle/data_raw/images/train/61/61_00121.jpg"
img = cv2.imread(image_path)
if img is None:
    print("Không load được ảnh")
    exit()

vehicles = detector_vehicle.detect(img)
for (x1, y1, x2, y2, conf, label) in vehicles:
    text = f"{label} {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    (tw, th), _ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.6, 2)

    if y1 - 10 < th:
        y_text = y1 + th + 5
    else:
        y_text = y1 - 10

    cv2.rectangle(img,(x1, y_text - th - 5),(x1 + tw, y_text + 5),(0,255,0),-1)
    
    cv2.putText(img, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


plates = detector_plate.detect(img)
warper = PlateWarper()
def sort_chars(chars):
    return sorted(chars, key=lambda c: c["bbox"][0])  # sort theo x

def decode_plate(chars):
    # chars = sort_chars(chars)
    return "".join([c["label"] for c in chars])

for i, plate in enumerate(plates):
    x1, y1, x2, y2 = plate["bbox"]
    pts = plate["points"]

    cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)

    if pts is not None:
        for (x, y) in pts:
            cv2.circle(img, (int(x), int(y)), 2, (0,0,255), -1)

        warped = warper.warp(img, pts)
        if warped is None:
            continue

        # warp_w = 320
        # warp_h = 96

        # warped = cv2.resize(warped, (warp_w, warp_h))
        h0, w0 = warped.shape[:2]

        warp_w = w0
        warp_h = h0

         # =============================
        # VỊ TRÍ HIỂN THỊ
        # =============================
        show_x = x1
        show_y = y1 - warp_h - 15

        # tránh vượt biên
        if show_y < 0:
            show_y = y2 + 15

        if show_x + warp_w > img.shape[1]:
            show_x = img.shape[1] - warp_w

        # =============================
        # PASTE LÊN FRAME
        # =============================
        img[
            show_y:show_y + warp_h,
            show_x:show_x + warp_w
        ] = warped

        # border thumbnail
        cv2.rectangle(img,
                      (show_x, show_y),
                      (show_x + warp_w, show_y + warp_h),
                      (255,255,255), 2)
        # =============================
        # DETECT KÝ TỰ
        # =============================
        chars = detector_char.detect(warped)
        plate_text = decode_plate(chars)

        # print("PLATE:", plate_text)

        # # =============================
        # # HIỂN THỊ TEXT NGAY DƯỚI WARP
        # # =============================
        text_x = show_x
        text_y = show_y + warp_h + 20

        # cv2.putText(img,
        #             plate_text,
        #             (text_x, text_y),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7,
        #             (0, 0, 255),2)
        plate_text = plate_text if plate_text else "UNKNOWN"

        (font_w, font_h), _ = cv2.getTextSize(
            plate_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        box_x1 = text_x
        box_y1 = text_y - font_h - 10
        box_x2 = text_x + font_w + 10
        box_y2 = text_y + 5

        # background box
        cv2.rectangle(img,
                    (box_x1, box_y1),
                    (box_x2, box_y2),
                    (0, 0, 255),  # đỏ giống plate
                    -1)

        # text
        cv2.putText(img,
                    plate_text,
                    (box_x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()