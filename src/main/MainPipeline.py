import cv2
import numpy as np
from PlateWarper import PlateWarper
from Filter_Plate import Filter_Plate
from PlateTrackingManager import PlateTrackingManager
import math
import json
from datetime import datetime
import os
from pathlib import Path 
import re
from collections import Counter

class MainPipeline:
    def __init__(self, vehicle_detector,vehicle_tracker, plate_detector, char_detector):
        # truyền instance đã khởi tạo từ ngoài vào
        self.vehicle_detector = vehicle_detector
        self.vehicle_tracker = vehicle_tracker
        self.plate_detector = plate_detector
        self.char_detector = char_detector
        self.frame_index = 0
        self.plate_manager = PlateTrackingManager( min_length=7, min_votes=3, max_history=20, expire_frames=120 )
        self.warper = PlateWarper(sharpness_threshold=150.0, sharpness_method='laplacian')
        self.sharp_eval = Filter_Plate(method="laplacian",char_threshold=80.0,plate_threshold=150.0
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

    def smooth_points(self, points_list_curr):
        #points_list_curr: list các np.array shape (4,2) hoặc tương đương
        if not self.prev_plate_points:
            # frame đầu tiên: chưa có gì để mượt
            self.prev_plate_points = [p.copy() for p in points_list_curr]
            return points_list_curr

        smoothed = []
        alpha = self.smooth_alpha

        #giả sử số lượng biển không đổi nhiều, matching theo index
        n = min(len(self.prev_plate_points), len(points_list_curr))
        for i in range(n):
            prev = self.prev_plate_points[i]
            curr = points_list_curr[i]
            smoothed_point = alpha * curr + (1 - alpha) * prev
            smoothed.append(smoothed_point)
        
        # nếu frame mới có nhiều hơn:
        for i in range(n, len(points_list_curr)):
            smoothed.append(points_list_curr[i])

        # cập nhật state cho frame sau
        self.prev_plate_points = [p.copy() for p in smoothed]
        return smoothed

    def prepare_plate_image(self, frame, bbox, pts):
        x1, y1, x2, y2 = bbox
        plate_crop = frame[y1:y2, x1:x2].copy()
        plate_img_for_char = plate_crop

        if pts is not None and self.warper.is_valid_plate(pts):
            warped = self.warper.warp(frame, pts)            
            if warped is not None and warped.size > 0:
                plate_img_for_char = warped

        sharpness = self.sharp_eval.measure_sharpness(plate_img_for_char)
        is_sharp_enough  = sharpness >= self.sharp_eval.plate_threshold
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
        print(text)
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
        for (x1, y1, x2, y2, conf, label) in vehicles:
            color = self.vehicle_colors.get(label, self.default_vehicle_color)
            # name  = self.vehicle_names.get(label, self.default_vehicle_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            #f"{label}
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)

    def draw_plate_pose_and_text(self, frame, bbox, pts, text, plate_img = None, char_boxes = None, scale = 2.0, offset = (10,-80)):
        x1, y1, x2, y2 = bbox
        # draw box plate
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)
        # draw keypoints
        if pts is not None:
            for (x, y) in pts:
                cv2.circle(frame, (int(x), int(y)), 2, (0,0,255), -1)

        # Check plate image
        if plate_img is None or plate_img.size == 0:
            return
        
        # Preview size
        bw = x2 - x1
        bh = y2 - y1
        preview_w = int(bw * scale)
        preview_h = int(bh * scale)

        preview = cv2.resize(plate_img,(preview_w, preview_h))

        # preview position
        offset = (bw // 2, -bh * 3)
        dx,dy = offset
        px1 = x1 + dx
        py1 = y1 + dy
        px2 = px1 + preview_w
        py2 = py1 + preview_h

        H, W = frame.shape[:2]
        if px1 < 0:
            px1 = 0
            px2 = preview_w

        if py1 < 0:
            py1 = 0
            py2 = preview_h

        if px2 > W:
            px2 = W
            px1 = W - preview_w

        if py2 > H:
            py2 = H
            py1 = H - preview_h

        preview_h_actual = py2 - py1
        preview_w_actual = px2 - px1

        preview = cv2.resize(preview,(preview_w_actual, preview_h_actual))

        #Paste preview
        frame[py1:py2, px1:px2] = preview
        cv2.rectangle(frame,(px1, py1),(px2, py2),(0, 0, 255),2)
        # if char_boxes:
        #     chars_sorted = sorted(char_boxes,key=lambda c: c["bbox"][0])
        #     char_text = "".join([c["label"] for c in chars_sorted])
        #     cv2.putText(frame,char_text,(px1, py2 + 25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2,cv2.LINE_AA)
        if text:
            cv2.putText(frame, text, (px1, py2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2, cv2.LINE_AA)

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
        Xử lý frame với tracked vehicles và ANPR
        Returns:
        tuple : (output_frame, results_list)
        """
        detections = self.vehicle_detector.detect(frame)
        vehicles =  self.vehicle_tracker.update(detections)

        for (track_id, x1, y1, x2, y2, conf, label) in vehicles:
            color = self.vehicle_colors.get(label, self.default_vehicle_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id} {label}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA)
            # Init track nếu chưa có   
            if track_id not in self.plate_manager.memory:
                self.plate_manager.init_track(track_id, vehicle_type=label, frame_index=self.frame_index)
            else:  #Cập nhật vehicle_type nếu đã tồn tại
                self.plate_manager.memory[track_id]["vehicle_type"] = label

            #self.track_memory[track_id]["last_seen"] = datetime.now().isoformat()
        
        # h,w = frame.shape[:2]
        plates = self.detect_plates(frame)
        # Map plates to vehicle
        plate_assigned = {}  # track_id -> plate info
        #results = []
        for plate in plates:
            bbox = plate["bbox"]
            pts = plate["points"]
            px1, py1, px2, py2 = bbox

            best_tid = None
            cx = (px1 + px2) // 2
            cy = (py1 + py2) // 2

            # Tìm vehicle chứa plate
            for ( track_id, vx1, vy1, vx2, vy2, vconf, vlabel ) in vehicles:
                inside = ( vx1 <= cx <= vx2 and vy1 <= cy <= vy2 )
                if inside: 
                    best_tid = track_id
                    break
            if best_tid is None:
                continue
            #Đánh dấu vehicle này có plate
            plate_assigned[best_tid] = {"bbox": bbox, "pts": pts}

            # prepare plate
            plate_img_for_char, plate_crop, sharpness, ok = self.prepare_plate_image(frame, bbox, pts)
            #Xử lý plate blur/unreadable
            if not ok or plate_img_for_char is None:
                self.plate_manager.update_unreadable_plate(
                    track_id    = best_tid,
                    frame_index = self.frame_index,
                    sharpness   = sharpness,
                    reason="blur"
                )
                continue

            # read 
            chars = self.char_detector.detect(plate_img_for_char)
            chars = self.sharp_eval.char_sharpness( plate_img_for_char, chars )  
            chars_for_text = [ c for c in chars if not c["is_blur"] ]

            if len(chars_for_text) < 4:
                self.plate_manager.update_unreadable_plate(
                    track_id    = best_tid,
                    frame_index = self.frame_index,
                    sharpness   = sharpness,
                    reason      = "insufficient_chars"
                )
                continue  

            raw_text = self.chars_to_text(chars_for_text)
            text = self.format_plate(raw_text)

            char_confs = [c.get("conf", 0)for c in chars_for_text]
            avg_conf = ( sum(char_confs) / len(char_confs) if char_confs else 0 )

            self.plate_manager.update_plate(
                track_id=best_tid,
                text=text,
                frame_index=self.frame_index,
                sharpness=sharpness,
                char_count=len(chars_for_text),
                avg_conf=avg_conf,
                bbox=bbox
            )
            # === Draw result ===
            stable_text = self.plate_manager.get_stable_text(best_tid)
            display_text = stable_text if stable_text else text
            # results.append({
            #     "track_id": best_tid,
            #     "text": text,
            #     "bbox": bbox
            # })
            self.draw_plate_pose_and_text(
                frame,
                bbox,
                pts,
                display_text,
                plate_img_for_char,
                char_boxes=chars
            )

        # Xử lý vehicles KHÔNG có plate
        for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
            if track_id not in plate_assigned:
                self.plate_manager.update_no_plate(
                track_id=track_id,
                frame_index=self.frame_index
            )
        #results từ plate_manager
        results = []
        for (track_id, vx1, vy1, vx2, vy2, vconf, vlabel) in vehicles:
            track_result = self.plate_manager.get_track_result(track_id)
            if track_result:
                results.append(track_result)

        return frame, results

    def convert_to_json(self,obj):
        # chuyển ndarray -> list
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # list/tuple -> xử lý từng phần tử
        if isinstance(obj, (list, tuple)):
            return [self.convert_to_json(x) for x in obj]
        # dict -> xử lý từng value
        if isinstance(obj, dict):
            return {k: self.convert_to_json(v) for k, v in obj.items()}
        return obj



# ===== CHẠY ẢNH / VIDEO =====

def run_on_image(image_path, pipeline, save_path=None, show=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không đọc được ảnh")

    out_img, results = pipeline.process_frame(img)

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

def run_tracker(video_source,pipeline,save_dir="./output",show=True,read_step=3):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_source}")

    # ===== tạo thư mục output =====
    os.makedirs(save_dir, exist_ok=True)
    video_name = Path(video_source).stem

    video_path = os.path.join(save_dir,f"{video_name}_tracked.mp4")
    json_path = os.path.join(save_dir,f"{video_name}_tracked.json")

    # ===== path unique =====
    unique_video_path = make_unique_path(video_path)
    unique_json_path = make_unique_path(json_path)

    # ===== video writer setup =====
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ===== writer =====
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(unique_video_path,fourcc,fps,(w, h))

    if not writer.isOpened():
        raise ValueError(
            f"Không tạo được video writer: {unique_video_path}"
        )

     # === Processing loop ===
    #best_by_id = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #  Update pipeline frame index
        pipeline.frame_index = frame_idx
        # ===== process =====
        #out_frame, results = pipeline.process_frame_with_tracked_vehicles(frame)

        # ===== OCR theo nhịp =====
        if frame_idx % read_step == 0:
            out_frame, results = pipeline.process_frame_with_tracked_vehicles(frame)
        else:
            #Frame không OCR: chỉ draw vehicles
            detections = pipeline.vehicle_detector.detect(frame)
            vehicles = pipeline.vehicle_tracker.update(detections)

            for (track_id, x1, y1, x2, y2, conf, label) in vehicles:
                if track_id not in pipeline.plate_manager.memory:
                    pipeline.plate_manager.init_track(
                        track_id, 
                        vehicle_type=label,
                        frame_index=frame_idx 
                    )
                color = pipeline.vehicle_colors.get(label, pipeline.default_vehicle_color)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id} {label}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2, cv2.LINE_AA)
                #Update last_seen_frame (không OCR nhưng vẫn track)
                if track_id in pipeline.plate_manager.memory:
                    pipeline.plate_manager.memory[track_id]["last_seen_frame"] = frame_idx
            out_frame = frame

        # Cleanup expired tracks mỗi 30 frames
        if frame_idx % 30 == 0 and frame_idx > 0:
            pipeline.plate_manager.remove_expired_tracks(frame_idx)

        # # ===== save video =====
        writer.write(out_frame)

        # ===== show =====
        if show:
            cv2.imshow("ANPR + Tracking", out_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    # ===== release =====
    cap.release()
    writer.release()

    if show:
        cv2.destroyAllWindows()

    #Finalize tất cả active tracks
    pipeline.plate_manager.finalize_all_active_tracks()
    #  Export đầy đủ từ plate_manager
    all_results = pipeline.plate_manager.export_all_results()
    with open(unique_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_source": video_source,
            "total_frames": frame_idx,
            "total_vehicles": len(all_results),
            "processing_params": {
                "read_step": read_step,
                "expire_frames": pipeline.plate_manager.expire_frames,
                "min_votes": pipeline.plate_manager.min_votes
            },
            "summary": {
                "verified": len([r for r in all_results if r["status"] == "verified"]),
                "readable": len([r for r in all_results if r["status"] == "readable"]),
                "low_confidence": len([r for r in all_results if r["status"] == "low_confidence"]),
                "unreadable": len([r for r in all_results if r["status"] == "unreadable"]),
                "no_plate": len([r for r in all_results if r["status"] == "no_plate"])
            },
            "vehicles": all_results
        }, f, ensure_ascii = False, indent=2)

    pipeline.plate_manager.clear_finalized()
    print(f"[INFO] Saved video: {unique_video_path}")
    print(f"[INFO] Saved JSON: {unique_json_path}")
    print(f"[INFO] Total vehicles: {len(all_results)}")
    print(f"[INFO] Status breakdown:")
    for status in ["verified", "readable", "low_confidence", "unreadable", "no_plate"]:
        count = len([r for r in all_results if r["status"] == status])
        print(f"  - {status}: {count}")
    # ===== save json =====

    return all_results

def make_unique_path(path):
    # nếu chưa tồn tại thì dùng luôn
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 1
    # thêm _1, _2, ... cho đến khi không trùng
    while True:
        new_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


def choose_better_plate(old, new):
    if old is None:
        return new
    if len(new.get("text", "")) > len(old.get("text", "")):
        return new
    return old
