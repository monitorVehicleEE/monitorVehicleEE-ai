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
import time

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
            'Q': '0',
            'D': '0',
            'A': '4',
            'L': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'G': '6'
        }
        self.digit_to_char = {
            '2': 'Z',
            '5': 'S',
            '8': 'B',
            '6': 'G',
            '4': 'A'
        }
        self.vehicle_colors = {
            "oto"         :(255, 0, 0),    # xanh dương
            "xe-tai"      : (0, 255, 255),  # vàng
            "xe-container": (0, 165, 255),  # cam
            "motorbike"   : (0, 255, 0),# xanh lá
        }
        self.default_vehicle_color = (255, 255, 255)
        self.vehicle_step = 2
        self.plate_step   = 6
        self.orc_step     = 8
        self.vehicle_sample_step = 3
        self.max_plate_frame_checks_per_cycle = 2
        self.max_ocr_reads_per_cycle = 3
        self.max_cached_frames = 30
        self.max_plate_sample_retries = 2

        self.last_detections = []
        self.last_vehicles = []
        self.has_vehicle_detection = False
        self.frame_cache = {}
        self.plate_detection_cache = {}
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

    def cache_frame(self, frame_index, frame):
        self.frame_cache[frame_index] = frame.copy()

    def get_cached_frame(self, frame_ref):
        return self.frame_cache.get(frame_ref)

    def prune_frame_caches(self):
        keep_refs = set()
        for mem in self.tracking_manager.memory.values():
            for sample in mem.get("best_vehicle_frames", []):
                frame_ref = sample.get("frame_ref")
                if frame_ref is not None:
                    keep_refs.add(frame_ref)
            for sample in mem.get("best_plate_frames", []):
                frame_ref = sample.get("frame_idx")
                if frame_ref is not None:
                    keep_refs.add(frame_ref)

        newest_refs = sorted(self.frame_cache.keys(), reverse=True)[:self.max_cached_frames]
        keep_refs.update(newest_refs)

        for frame_ref in list(self.frame_cache.keys()):
            if frame_ref not in keep_refs:
                self.frame_cache.pop(frame_ref, None)
                self.plate_detection_cache.pop(frame_ref, None)

    def detect_plates_cached(self, frame_ref, frame):
        if frame_ref not in self.plate_detection_cache:
            t0 = time.perf_counter()
            plates = self.detect_plates(frame)
            plate_detect_time = time.perf_counter() - t0
            print(f"[Plate Detect] {plate_detect_time*1000:.1f} ms")
            self.plate_detection_cache[frame_ref] = plates
        return self.plate_detection_cache[frame_ref]

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

    def draw_plate_pose_and_text( self, frame, vehicle_bbox, plate_bbox, pts, text, plate_img=None, char_boxes=None ):
        """
        Vẽ preview biển số bên trong bbox vehicle
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        px1, py1, px2, py2 = plate_bbox
       
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
        # Vẽ preview trong bbox vehicle
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


    def estimate_plate_angle_deg(pts):
        """
        Trả về góc nghiêng (độ) theo trục ngang.
        """
        if pts is None or len(pts) < 2:
            return 0.0

        # giả sử pts[0] = top-left, pts[1] = top-right
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return 90.0

        angle_rad = math.atan2(dy, dx)   # góc so với trục x
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    
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
            # self.frame_index += 1
            total_start = time.perf_counter()
            H, W = frame.shape[:2]
            raw_frame = frame.copy()

            run_vehicle_detection = (
                self.frame_index % self.vehicle_step == 0
                or not self.has_vehicle_detection
            )
            should_sample_vehicle_frame = ( self.frame_index % self.vehicle_sample_step == 0 )
            if should_sample_vehicle_frame:
                self.cache_frame(self.frame_index, raw_frame)

            if run_vehicle_detection:
                t0 = time.perf_counter()
                detections = self.vehicle_detector.detect(raw_frame)
                vehicle_detect_time = time.perf_counter() - t0
                print(f"[Vehicle Detect] {vehicle_detect_time*1000:.1f} ms")
                vehicles = self.vehicle_tracker.update(detections)
                self.last_vehicles = vehicles
                self.has_vehicle_detection = True
            else:
                vehicles = self.last_vehicles

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
                mem["vehicle_confidence"] = vconf
                mem["vehicle_bbox"] = (vx1, vy1, vx2, vy2)
                mem["last_seen_frame"] = self.frame_index
                mem["last_seen_time"] = datetime.now().isoformat()

                if should_sample_vehicle_frame:
                    vx1c, vy1c = max(0, vx1), max(0, vy1)
                    vx2c, vy2c = min(W, vx2), min(H, vy2)
                    if vx2c <= vx1c or vy2c <= vy1c:
                        continue

                    vehicle_roi = raw_frame[vy1c:vy2c, vx1c:vx2c]
                    if vehicle_roi.size == 0:
                        continue

                    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                    area = max(0, (vx2c - vx1c) * (vy2c - vy1c))
                    priority = sharp + 0.01 * area

                    self.tracking_manager.update_best_vehicle_frame(
                        track_id=track_id,
                        frame_index=self.frame_index,
                        bbox=(vx1c, vy1c, vx2c, vy2c),
                        sharpness=sharp,
                        priority=priority,
                        vehicle_img=vehicle_roi.copy(),
                        frame_ref=self.frame_index,
                        max_samples=6
                    )
            
            # 2. Phát hiện biển số theo chu kỳ plate_step
            if self.frame_index % self.plate_step == 0:
                plate_frame_checks = 0
                for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
                    if plate_frame_checks >= self.max_plate_frame_checks_per_cycle:
                        break
                    mem = self.tracking_manager.memory.get(track_id, {})
                    if mem.get("best_text") and mem.get("best_count", 0) >= self.tracking_manager.min_votes:
                        continue

                    top_samples = self.tracking_manager.get_best_vehicle_samples(
                        track_id,
                        top_k=3
                    )
                    for sample in top_samples:
                        if plate_frame_checks >= self.max_plate_frame_checks_per_cycle:
                            break
                        if sample.get("plate_checked"):
                            continue
                        if sample.get("plate_attempts", 0) >= self.max_plate_sample_retries:
                            sample["plate_checked"] = True
                            continue
                        if self.frame_index < sample.get("retry_after_frame", -1):
                            continue

                        sample_track_id = sample.get("track_id", track_id)
                        if sample_track_id not in self.tracking_manager.memory:
                            sample["plate_checked"] = True
                            continue

                        frame_ref = sample.get("frame_ref", sample.get("frame_idx"))
                        scene_frame = (
                            self.get_cached_frame(frame_ref)
                            if frame_ref is not None
                            else None
                        )
                        if scene_frame is None or scene_frame.size == 0:
                            sample["plate_checked"] = True
                            continue

                        sample["plate_attempts"] = sample.get("plate_attempts", 0) + 1
                        sample["retry_after_frame"] = self.frame_index + self.plate_step
                        plate_frame_checks += 1

                        vehicle_bbox = sample["bbox"]
                        vx1s, vy1s, vx2s, vy2s = vehicle_bbox

                        if frame_ref is not None:
                            plates = self.detect_plates_cached(frame_ref, scene_frame)
                        else:
                            t0 = time.perf_counter()
                            plates = self.detect_plates(scene_frame)
                            plate_detect_time = time.perf_counter() - t0
                            print(f"[Plate Detect] {plate_detect_time*1000:.1f} ms")

                        found_plate_for_sample = False
                        for plate in plates:
                            bbox = plate["bbox"]
                            pts = plate["points"]
                            px1, py1, px2, py2 = bbox
                            cx = (px1 + px2) / 2.0
                            cy = (py1 + py2) / 2.0
                            if not (vx1s <= cx <= vx2s and vy1s <= cy <= vy2s):
                                continue

                            plate_img_for_char, plate_crop, sharpness, ok = \
                                self.prepare_plate_image(scene_frame, bbox, pts)
                            area = max(0, (px2 - px1) * (py2 - py1))
                            priority = sharpness + 0.01 * area

                            if not ok or plate_img_for_char is None:
                                self.tracking_manager.update_unreadable_plate(
                                    track_id=sample_track_id,
                                    frame_index=self.frame_index,
                                    sharpness=sharpness,
                                    reason="blur_or_cut"
                                )
                                continue

                            found_plate_for_sample = True
                            sample["plate_checked"] = True
                            self.tracking_manager.update_best_plate_frame(
                                track_id=sample_track_id,
                                frame_index=sample["frame_idx"],
                                bbox=bbox,
                                sharpness=sharpness,
                                priority=priority,
                                plate_img=plate_img_for_char,
                                pts=pts,
                                vehicle_bbox=vehicle_bbox,
                                max_samples=8
                            )

                        if not found_plate_for_sample and sample.get("plate_attempts", 0) >= self.max_plate_sample_retries:
                            sample["plate_checked"] = True

            # 3. Chạy nhận diện ký tự (OCR) theo chu kỳ orc_step
            if self.frame_index % self.orc_step == 0:
                for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
                    best_samples = self.tracking_manager.get_best_plate_samples(track_id, top_k=8)
                    if not best_samples:
                        continue
                    for best_sample in best_samples:
                        if best_sample.get("ocr_failed"):
                            continue

                        bbox = best_sample["bbox"]
                        plate_img_for_char = best_sample.get("plate_img")

                        if plate_img_for_char is None:
                            best_sample["ocr_failed"] = True
                            continue
                            
                        t0 = time.perf_counter()

                        chars = self.char_detector.detect(plate_img_for_char)

                        ocr_time = time.perf_counter() - t0

                        print(f"[OCR] {ocr_time*1000:.1f} ms")
                        chars = self.filter_plate.char_sharpness(plate_img_for_char, chars)
                        chars_for_text = chars

                        if len(chars_for_text) < 4:
                            best_sample["ocr_failed"] = True
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
                        before_successful_reads = self.tracking_manager.memory.get(
                            track_id, {}
                        ).get("successful_reads", 0)

                        self.tracking_manager.update_plate(
                            track_id=track_id,
                            text=text,
                            frame_index=self.frame_index,
                            sharpness=best_sample["sharpness"],
                            char_count=len(chars_for_text),
                            avg_conf=avg_conf,
                            bbox=bbox
                        )
                        after_successful_reads = self.tracking_manager.memory.get(
                            track_id, {}
                        ).get("successful_reads", 0)
                        if after_successful_reads <= before_successful_reads:
                            best_sample["ocr_failed"] = True
                            continue

                        best_sample["ocr_success"] = True
                        best_sample["ocr_text"] = text
                        best_sample["ocr_confidence"] = avg_conf
                        best_sample["ocr_score"] = (
                            best_sample["sharpness"]
                            + avg_conf * 50.0
                            + len(chars_for_text) * 2.0
                        )

                        char_avg_sharp = self.filter_plate.avg_sharpness(chars_for_text, ignore_blur=False)
                        quality_score = best_sample["sharpness"] + char_avg_sharp * 0.3 + avg_conf * 50.0
                        self.tracking_manager.update_after_anpr(
                            track_id, self.frame_index, quality_score
                        )
                        stable_text = self.tracking_manager.get_stable_text(track_id)
                        display_text = stable_text if stable_text else text
                        self.tracking_manager.update_display_info(track_id, display_text)
                        break

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
                    draw_start = time.perf_counter()
                    self.draw_plate_pose_and_text(
                        frame,
                        current_vehicle_bbox,
                        best_sample["bbox"], 
                        best_sample.get("pts"),
                        display_text,
                        best_sample.get("plate_img"),
                        char_boxes=None # Có thể truyền list char nếu cần vẽ từng ký tự cụ thể
                    )
                    draw_time = time.perf_counter() - draw_start
                    print(f"[DRAW] {draw_time*1000:.1f} ms")

            # # 5. Dọn dẹp và kết xuất kết quả danh sách tracking
            self.tracking_manager.remove_expired_tracks(self.frame_index)
            self.prune_frame_caches()
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

            total_time = time.perf_counter() - total_start

            fps = 1.0 / total_time if total_time > 0 else 0

            print(f"[TOTAL] {total_time*1000:.1f} ms | FPS: {fps:.1f}")
            return frame, results
