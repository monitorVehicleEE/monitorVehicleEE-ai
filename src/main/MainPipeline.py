import cv2
import numpy as np
from src.main.PlateWarper import PlateWarper
from src.main.Filter_Plate import Filter_Plate
from src.main.TrackingManager import TrackingManager
import math
import json
from datetime import datetime
import os
from pathlib import Path
import re
from collections import defaultdict, Counter
import random


class MainPipeline:
    def __init__(self, vehicle_detector,vehicle_tracker, plate_detector, char_detector):
        # truyền instance đã khởi tạo từ ngoài vào
        self.vehicle_detector = vehicle_detector
        self.vehicle_tracker = vehicle_tracker
        self.plate_detector = plate_detector
        self.char_detector = char_detector
        self.frame_index = 0
        self.tracking_manager = TrackingManager( min_length=7, min_votes=3, max_history=20, expire_frames=120 )
        self.warper = PlateWarper(sharpness_threshold=150.0, sharpness_method='laplacian')
        self.filter_plate = Filter_Plate(method="laplacian",char_threshold=80.0,plate_threshold=150.0
        )
        self.prev_plate_points = []
        self.smooth_alpha = 0.7
        self.char_to_digit = {
            'O': '0',
            'Q': '0',
            'D': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'G': '6'
        }
        self.digit_to_char = {
            '0': 'O',
            '1': 'I',
            '2': 'Z',
            '5': 'S',
            '8': 'B',
            '6': 'G'
        }
        self.vehicle_colors = {
            "oto"         :(255, 0, 0),    # xanh dương
            "xe-tai"      : (0, 255, 255),  # vàng
            "xe-container": (0, 165, 255),  # cam
            "motorbike"   : (0, 255, 0),# xanh lá
        }
        self.default_vehicle_color = (255, 255, 255)
        self.vehicle_step = 3
        self.plate_step   = 5
        self.orc_step     = 8

        self.v_count      = 0
        self.p_count      = 0
        self.o_count      = 0
        self.last_detections = []
        # self.vehicle_names = {
        #     "oto":        "Ô tô",
        #     "xe-tai":     "Xe tải",
        #     "container":  "Xe container",
        #     "motorbike":  "Xe máy",
        # }
        # self.default_vehicle_name = "Phương tiện"


    def detect_vehicles(self, frame):
        return self.vehicle_detector.detect(frame)

    def detect_plates(self, frame):
        return self.plate_detector.detect(frame)

    def prepare_plate_image(self, frame, bbox, pts):
        x1, y1, x2, y2 = bbox
        plate_crop = frame[y1:y2, x1:x2].copy()
        plate_img_for_char = plate_crop

        if pts is not None and self.warper.is_valid_plate(pts):
            warped = self.warper.warp(frame, pts)            
            if warped is not None and warped.size > 0:
                plate_img_for_char = warped


        sharpness = self.filter_plate.measure_sharpness(plate_img_for_char)
        is_sharp_enough  = sharpness >= self.filter_plate.plate_threshold
        if not is_sharp_enough:
            return None, plate_crop, sharpness, False
        return plate_img_for_char, plate_crop, sharpness, True


    def chars_to_text(self, chars, line_merge_ratio=0.6):
        if not chars:
            return ""

        # Tính center x,y và chiều cao cho mỗi char
        def bbox_info(c):
            x1, y1, x2, y2 = c["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            h = (y2 - y1)
            return cx, cy, h

        # Sort theo cy để có thứ tự từ trên xuống
        chars = sorted(chars, key=lambda c: bbox_info(c)[1])

        # Ước lượng chiều cao trung bình -> dùng làm ngưỡng group theo dòng
        hs = [bbox_info(c)[2] for c in chars]
        h_avg = np.mean(hs) if hs else 0
        if h_avg <= 0:
            h_avg = 20  # fallback nhỏ

        # Ngưỡng cho phép lệch y trong cùng 1 dòng
        # line_merge_ratio: 0.5–0.7 là ổn, biển nghiêng thì tăng lên
        y_threshold = h_avg * line_merge_ratio
        lines = []

        for c in chars:
            cx, cy, h = bbox_info(c)
            placed = False

            for line in lines:
                line_cys = [bbox_info(x)[1] for x in line]
                cy_mean = np.mean(line_cys)
                if abs(cy - cy_mean) <= y_threshold:
                    line.append(c)
                    placed = True
                    break

            if not placed:
                lines.append([c])


        # Sort các dòng theo Y (trên -> dưới)
        lines = sorted(lines, key=lambda line: np.mean([bbox_info(c)[1] for c in line]))

        # Sort trong từng dòng theo X (trái -> phải)
        line_texts = []
        for line in lines:
            line_sorted = sorted(line, key=lambda c: bbox_info(c)[0])
            text = "".join([c["label"] for c in line_sorted])
            line_texts.append(text)


        plate_text = "\n".join(line_texts)
        return plate_text

    def  format_plate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        lines = [line.strip().upper() for line in text.split("\n") if line.strip()]
        lines = [line for line in lines if line]
       
        if not lines:
            return ""

        # Trường hợp đặc biệt: Biển nước ngoài hệ 3 dòng hoặc định dạng dài (ví dụ: 80-NG-123-45)
        # Ta gộp tạm thành 1 chuỗi để check xem có chứa cụm từ nước ngoài không
        full_raw = "".join(lines)
        for special_code in ["NG", "QT", "NN"]:
            if special_code in full_raw:
                # Sửa lỗi nhận diện nhầm chữ/số cho cụm đặc biệt trước khi format
                # Ép 2 ký tự đầu làm số, cụm chữ giữ nguyên, các ký tự sau làm số
                match = re.match(r"^([A-Z0-9]{2})(" + special_code + r")([A-Z0-9]{5})$", full_raw)
                if match:
                    prefix = match.group(1)
                    suffix = match.group(3)
                    # Ép tiền tố về số
                    prefix = "".join([self.char_to_digit.get(c, c) for c in prefix])
                    # Ép hậu tố về số
                    suffix = "".join([self.char_to_digit.get(c, c) for c in suffix])
                    if len(suffix) == 5:
                        return f"{prefix}-{special_code}-{suffix[:3]}.{suffix[3:]}"


        if len(lines) == 2:
            line1, line2 = lines[0], lines[1]
            l1_chars = list(line1)
            l2_chars = list(line2)
            for i in range(min(2, len(l1_chars))):
                if l1_chars[i].isalpha() and l1_chars[i] in self.char_to_digit:
                    l1_chars[i] = self.char_to_digit[l1_chars[i]]
            line1 = "".join(l1_chars)
            # Kiểm tra bẫy lỗi phân dòng (Không có ký tự thứ 3 ở dòng 1)
            if len(line1) == 2 and line1.isdigit() and len(line2) > 0 and not line2[0].isdigit():
                line1 = line1 + line2[0]
                line2 = line2[1:]
            # Kiểm tra và xử lý ký tự thứ 3 của line1
            l1_chars = list(line1)
            if len(l1_chars) >= 3:
                c3 = l1_chars[2]
                if c3.isdigit() and c3 in self.digit_to_char:
                    l1_chars[2] = self.digit_to_char[c3]
                line1 = "".join(l1_chars)
           
            l2_chars = list(line2)
            for i in range(len(l2_chars)):
                if l2_chars[i].isalpha() and l2_chars[i] in self.char_to_digit:
                    l2_chars[i] = self.char_to_digit[l2_chars[i]]
            line2 = "".join(l2_chars)


            # if len(line2) > 5:
            #     # Số ký tự dư so với 5
            #     extra = len(line2) - 5
            #     # Đẩy 'extra' ký tự đầu của line2 lên cuối line1
            #     move_part = line2[:extra]
            #     line1 = line1 + move_part
            #     line2 = line2[extra:]
            # Kiểm tra số lượng số đuôi của line2 để chèn dấu format
            if len(line2) == 5 and line2.isdigit():
                return f"{line1}-{line2[:3]}.{line2[3:]}"
            elif len(line2) == 4 and line2.isdigit():
                return f"{line1}-{line2}"
            else:
                return f"{line1}-{line2}"
        else:
            text = lines[0]
            # # Kiểm tra biển quân đội (Ví dụ: AA1234 hoặc BB12345)
            if re.match(r"^[A-Z]{2}\d{4,5}$", text):
                if len(text) == 6: # Dạng AA1234
                    return f"{text[:2]}-{text[2:]}"
                else: # Dạng BB12345 -> BB-123.45
                    return f"{text[:2]}-{text[2:5]}.{text[5:]}"
            match = re.match(r"^(\d{2}[A-Z]{1,2})(\d+)$", text)
            if match:
                header, body = match.group(1), match.group(2)
           
                if len(body) == 5:
                    # Biển 5 số: 30A12345 -> 30A-123.45
                    return f"{header}-{body[:3]}.{body[3:]}"
                elif len(body) == 4:
                    # Biển 4 số cũ: 29M1234 -> 29M-1234
                    return f"{header}-{body}"
                else:
                    # Fallback nếu số lượng số lạ
                    return f"{header}-{body}"


            return text


    def draw_vehicles(self, frame, vehicles):
        for (track_id, x1, y1, x2, y2, conf, label) in vehicles:
            color = self.vehicle_colors.get(label, self.default_vehicle_color)
            # name  = self.vehicle_names.get(label, self.default_vehicle_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #f"{label}
            cv2.putText(frame, f"ID:{track_id} --{label} -- {conf:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)

    # def draw_plate_pose_and_text(self, frame, bbox, pts, text, plate_img = None, char_boxes = None, scale = 2.0, offset = (10,-80)):
    #     x1, y1, x2, y2 = bbox
    #     # draw box plate
    #     cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)
    #     # draw keypoints
    #     if pts is not None:
    #         for (x, y) in pts:
    #             cv2.circle(frame, (int(x), int(y)), 2, (0,0,255), -1)


    #     # Check plate image
    #     if plate_img is None or plate_img.size == 0:
    #         return
       
    #     # Preview size
    #     bw = x2 - x1
    #     bh = y2 - y1
    #     preview_w = int(bw * scale)
    #     preview_h = int(bh * scale)


    #     preview = cv2.resize(plate_img,(preview_w, preview_h))


    #     # preview position
    #     offset = (bw // 2, -bh * 3)
    #     dx,dy = offset
    #     px1 = x1 + dx
    #     py1 = y1 + dy
    #     px2 = px1 + preview_w
    #     py2 = py1 + preview_h


    #     H, W = frame.shape[:2]
    #     if px1 < 0:
    #         px1 = 0
    #         px2 = preview_w


    #     if py1 < 0:
    #         py1 = 0
    #         py2 = preview_h


    #     if px2 > W:
    #         px2 = W
    #         px1 = W - preview_w


    #     if py2 > H:
    #         py2 = H
    #         py1 = H - preview_h


    #     preview_h_actual = py2 - py1
    #     preview_w_actual = px2 - px1


    #     preview = cv2.resize(preview,(preview_w_actual, preview_h_actual))


    #     #Paste preview
    #     frame[py1:py2, px1:px2] = preview
    #     cv2.rectangle(frame,(px1, py1),(px2, py2),(0, 0, 255),2)
    #     # if char_boxes:
    #     #     chars_sorted = sorted(char_boxes,key=lambda c: c["bbox"][0])
    #     #     char_text = "".join([c["label"] for c in chars_sorted])
    #     #     cv2.putText(frame,char_text,(px1, py2 + 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,cv2.LINE_AA)
    #     if text:
    #         cv2.putText(frame, text, (px1, py2 + 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9,
    #             (0, 255, 0), 2, cv2.LINE_AA)

    def draw_plate_pose_and_text( self, frame, vehicle_bbox, plate_bbox, pts, text, plate_img=None, char_boxes=None ):
        """
        Vẽ preview biển số bên trong bbox vehicle
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        px1, py1, px2, py2 = plate_bbox
        # ===== Draw plate bbox =====
        # cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 1)
        # # ===== Draw plate keypoints =====
        # if pts is not None:
        #     for (x, y) in pts:
        #         cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        # ===== Draw points on preview =====
       
        # ===== Không có ảnh biển =====
        if plate_img is None or plate_img.size == 0:
            return
        vehicle_w = vx2 - vx1
        vehicle_h = vy2 - vy1
        # ===== Preview size =====
        preview_w = int(vehicle_w * 0.45)

        aspect = plate_img.shape[0] / plate_img.shape[1]
        preview_h = int(preview_w * aspect)

        # ===== Giới hạn nếu quá to =====
        max_preview_h = int(vehicle_h * 0.35)

        if preview_h > max_preview_h:
            preview_h = max_preview_h
            preview_w = int(preview_h / aspect)

        preview = cv2.resize(plate_img, (preview_w, preview_h))

        # =========================================================
        # Vẽ preview trong bbox vehicle (góc trên phải)
        # =========================================================

        margin = 10

        draw_x1 = vx2 - preview_w - margin
        draw_y1 = vy1 + margin

        draw_x2 = draw_x1 + preview_w
        draw_y2 = draw_y1 + preview_h

        H, W = frame.shape[:2]

        # ===== Clamp =====
        draw_x1 = max(0, draw_x1)
        draw_y1 = max(0, draw_y1)

        draw_x2 = min(W, draw_x2)
        draw_y2 = min(H, draw_y2)

        actual_w = draw_x2 - draw_x1
        actual_h = draw_y2 - draw_y1

        if actual_w <= 0 or actual_h <= 0:
            return

        preview = cv2.resize(preview, (actual_w, actual_h))
        if pts is not None:
            scale_x = actual_w / (px2 - px1)
            scale_y = actual_h / (py2 - py1)

            for (x, y) in pts:
                rx = int((x - px1) * scale_x)
                ry = int((y - py1) * scale_y)

                cv2.circle(
                    preview,
                    (rx, ry),
                    1,
                    (0, 0, 255),
                    -1
                )
                cv2.rectangle(
                    frame,
                    (draw_x1, draw_y1),
                    (draw_x2, draw_y2),
                    (0, 0, 255),
                    1
                )


        # ===== Paste =====
        frame[draw_y1:draw_y2, draw_x1:draw_x2] = preview

        # Border preview
        cv2.rectangle(
            frame,
            (draw_x1, draw_y1),
            (draw_x2, draw_y2),
            (0, 0, 255),
            2
        )

        # ===== Text =====
        if text:
            text_y = draw_y2 + 25

            # nếu text bị tràn xuống ngoài vehicle
            if text_y > vy2 - 5:
                text_y = draw_y1 - 10

            cv2.putText(
                frame,
                text,
                (draw_x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    # ===== HÀM CHÍNH =====
    # với mỗi plate bbox tìm vehicle bbox nào bao phủ plate rồi lấy track_id đó
    def bbox_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])


        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH


        if interArea == 0:
            return 0.0
       
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_frame_with_tracked_vehicles(self, frame):
        """
        Xử lý frame với tracked vehicles, dùng adaptive ANPR + voting
        """
        self.frame_index += 1
        H, W = frame.shape[:2]

        detections = self.vehicle_detector.detect(frame)
        vehicles = self.vehicle_tracker.update(detections)

        draw_list = [(track_id, vx1, vy1, vx2, vy2, vconf, vlabel) for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles]
        self.draw_vehicles(frame, draw_list)

        # 1. Cập nhật thông tin cơ bản cho các phương tiện đang được track
        for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
            if track_id not in self.tracking_manager.memory:
                self.tracking_manager.init_track(
                    track_id,
                    vehicle_type=vlabel,
                    frame_index=self.frame_index
                )
            mem = self.tracking_manager.memory[track_id]
            mem["vehicle_type"] = vlabel
            mem["last_seen_frame"] = self.frame_index
            mem["last_seen_time"] = datetime.now().isoformat()
        
        # 2. Phát hiện biển số theo chu kỳ plate_step
        if self.frame_index % self.plate_step == 0:
            plates = self.detect_plates(frame)
            for plate in plates:
                bbox = plate["bbox"]      # (px1,py1,px2,py2)
                pts  = plate["points"]
                px1, py1, px2, py2 = bbox

                best_tid = None
                cx = (px1 + px2) / 2.0
                cy = (py1 + py2) / 2.0
                for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
                    inside = (vx1 <= cx <= vx2) and (vy1 <= cy <= vy2)
                    if inside:
                        best_tid = track_id
                        break

                if best_tid is None:
                    continue

                plate_img_for_char, plate_crop, sharpness, ok = self.prepare_plate_image(frame, bbox, pts)
                x1, y1, x2, y2 = bbox
                area = max(0, (x2 - x1) * (y2 - y1))
                priority = sharpness + 0.01 * area
                
                self.tracking_manager.update_best_plate_frame(
                        track_id=best_tid,
                        frame_index=self.frame_index,
                        bbox=bbox,
                        sharpness=sharpness,
                        priority=priority,
                        plate_img=plate_img_for_char if ok else None,
                        pts=pts,
                        vehicle_bbox=(vx1, vy1, vx2, vy2),
                        max_samples=5
                    )
                if not ok or plate_img_for_char is None:
                        self.tracking_manager.update_unreadable_plate(
                            track_id=best_tid,
                            frame_index=self.frame_index,
                            sharpness=sharpness,
                            reason="blur"
                        )
                        continue

        # 3. Chạy nhận diện ký tự (OCR) theo chu kỳ orc_step
        if self.frame_index % self.orc_step == 0:
            for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
                best_samples = self.tracking_manager.get_best_plate_samples(track_id, top_k=5)
                if not best_samples:
                    continue
                best_sample = best_samples[0]
                bbox = best_sample["bbox"]
                plate_img_for_char = best_sample.get("plate_img")

                if plate_img_for_char is None:
                    continue
                    
                chars = self.char_detector.detect(plate_img_for_char)
                chars = self.filter_plate.char_sharpness(plate_img_for_char, chars)
                chars_for_text = chars

                if len(chars_for_text) < 4:
                    self.tracking_manager.update_unreadable_plate(
                        track_id=track_id,
                        frame_index=self.frame_index,
                        sharpness=best_sample["sharpness"],
                        reason="insufficient_chars"
                    )
                    quality_score = best_sample["sharpness"]
                    self.tracking_manager.update_after_anpr(
                        track_id, self.frame_index, quality_score
                    )
                    continue

                raw_text = self.chars_to_text(chars_for_text)
                text = self.format_plate(raw_text)

                char_confs = [c.get("conf", 0) for c in chars_for_text]
                avg_conf = (sum(char_confs) / len(char_confs) if char_confs else 0)

                self.tracking_manager.update_plate(
                    track_id=track_id,
                    text=text,
                    frame_index=self.frame_index,
                    sharpness=best_sample["sharpness"],
                    char_count=len(chars_for_text),
                    avg_conf=avg_conf,
                    bbox=bbox
                )

                char_avg_sharp = self.filter_plate.avg_sharpness(chars_for_text, ignore_blur=False)
                quality_score = best_sample["sharpness"] + char_avg_sharp * 0.3 + avg_conf * 50.0
                self.tracking_manager.update_after_anpr(
                    track_id, self.frame_index, quality_score
                )
                stable_text = self.tracking_manager.get_stable_text(track_id)
                display_text = stable_text if stable_text else text
                self.tracking_manager.update_display_info(track_id, display_text)

        # 4. RENDERING ZONE: Vẽ đồ họa lên từng khung hình (Chạy ở MỌI frame)
        for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
            disp = self.tracking_manager.get_display_info(track_id)
            if not disp or not disp.get('text'):
                continue # Bỏ qua, không vẽ nếu chưa đọc được chữ / biển không đọc được
            
            display_text = disp['text']

            best_samples = self.tracking_manager.get_best_plate_samples(track_id, top_k=1)
            if not best_samples:
                continue
            
            best_sample = best_samples[0]
           # disp = self.tracking_manager.get_display_info(track_id)
            display_text = disp['text'] 

            # Lấy thông tin bounding box hiện tại của phương tiện để đồng bộ vị trí vẽ
            current_vehicle_bbox = (vx1, vy1, vx2, vy2)

            if best_sample.get("plate_img") is not None:
                self.draw_plate_pose_and_text(
                    frame,
                    current_vehicle_bbox,
                    best_sample["bbox"], 
                    best_sample.get("pts"),
                    display_text,
                    best_sample.get("plate_img"),
                    char_boxes=None # Có thể truyền list char nếu cần vẽ từng ký tự cụ thể
                )

        # 5. Dọn dẹp và kết xuất kết quả danh sách tracking
        self.tracking_manager.remove_expired_tracks(self.frame_index)
        results = []
        for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
            track_result = self.tracking_manager.get_track_result(track_id)
            if track_result:
                results.append(track_result)

        # 6. In danh sách tổng hợp góc trên bên trái màn hình
        y_offset = 100
        for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
            disp = self.tracking_manager.get_display_info(track_id)
            if not disp:
                continue

            label_str = f"ID:{track_id}| {vlabel} | {disp['text']}"
            cv2.putText(
                frame,
                label_str,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            y_offset += 30

        return frame, results

def run_on_image(image_path, pipeline, save_path=None, show=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không đọc được ảnh")

    out_img, results = pipeline.process_frame_with_tracked_vehicles(img)

    if save_path is not None:
        # nếu save_path là folder -> tự tạo tên file
        if os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
            base = os.path.basename(image_path)          
            name, ext = os.path.splitext(base)         
            out_path = os.path.join(save_path, name + "_out5" + ext)
        else:
            # save_path đã là full path tới file
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_path = save_path

        # ok = cv2.imwrite(out_path, out_img)
        # if not ok:
        #     print("[-] Lưu ảnh thất bại tại:", out_path)

    if show:
        cv2.imshow("Result", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_img, results