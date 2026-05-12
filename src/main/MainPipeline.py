import cv2
import numpy as np
from PlateWarper import PlateWarper
import math


class MainPipeline:
    def __init__(self, vehicle_detector, plate_detector, char_detector):
        # truyền instance đã khởi tạo từ ngoài vào
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.char_detector = char_detector
        self.warper = PlateWarper()
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
                #plate_img_for_char = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                # return plate_img_for_char, plate_crop
                
        return plate_img_for_char, plate_crop

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

        # Nếu chỉ muốn 1 string liền (biển 2 dòng vẫn đọc liền)
        plate_text = "".join(line_texts)
        # Nếu muốn giữ xuống dòng: "\n".join(line_texts)

        return plate_text

    # def format_plate(self, text: str) -> str:
    #     """
    #     - 43AB12345 -> 43AB-123.45
    #     - 43E12345  -> 43E-123.45
    #     - 43AB1234  -> 43AB-1234
    #     - 43E1234   -> 43E-1234
    #     """
    #     t = text.replace(" ", "")
    #     if len(t) < 4:
    #         return t

    #     # lấy phần số liên tục ở cuối
    #     tail_digits = ""
    #     i = len(t) - 1
    #     while i >= 0 and t[i].isdigit():
    #         tail_digits = t[i] + tail_digits
    #         i -= 1

    #     head = t[:i+1]          # phần chữ + mã tỉnh
    #     nums = tail_digits      # phần số

    #     if len(nums) == 5:
    #         # biển 5 số: xxx-123.45
    #         group3 = nums[:3]
    #         group2 = nums[3:]
    #         return f"{head}-{group3}.{group2}"
    #     elif len(nums) == 4:
    #         # biển 4 số: xxx-1234 (không chấm)
    #         return f"{head}-{nums}"
    #     else:
    #         # các trường hợp khác: giữ nguyên, chỉ thêm '-' nếu có head và nums
    #         if head and nums:
    #             return f"{head}-{nums}"
    #         return t

    def format_plate(self, text: str) -> str:
        t = text.replace(" ", "")
        n = len(t)
        if n < 5:
            return t

        # Trường hợp 9 ký tự: ví dụ 43D129560 -> 43D1-295.60
        if n == 9:
            head = t[:-5]          # 43D1
            nums = t[-5:]          # 29560
            group3 = nums[:3]      # 295
            group2 = nums[3:]      # 60
            return f"{head}-{group3}.{group2}"

        # Trường hợp 8 ký tự: ví dụ 43A60921 -> 43A-609.21
        if n == 8:
            head = t[:-5]          # 43A
            nums = t[-5:]          # 60921
            group3 = nums[:3]      # 609
            group2 = nums[3:]      # 21
            return f"{head}-{group3}.{group2}"

        # Các trường hợp chung: fallback dùng tail digits như cũ
        tail_digits = ""
        i = n - 1
        while i >= 0 and t[i].isdigit():
            tail_digits = t[i] + tail_digits
            i -= 1

        head = t[:i+1]
        nums = tail_digits

        if len(nums) == 5:
            group3 = nums[:3]
            group2 = nums[3:]
            return f"{head}-{group3}.{group2}"
        elif len(nums) == 4:
            return f"{head}-{nums}"
        else:
            if head and nums:
                return f"{head}-{nums}"
            return t

    def binding_char(self, text: str) -> str:
        if len(text) < 3:
            return text
        
        chars = list(text)
        c3 = chars[2]

        if c3.isdigit() and c3 in self.digit_to_char:
            chars[2] = self.digit_to_char[c3]

        for i in range(4, len(chars)):
            c = chars[i]
            if c.isalpha() and c in self.char_to_digit:
                chars[i] = self.char_to_digit[c]
                
        return "".join(chars)

    def recognize_plate_text(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return "", []
        chars = self.char_detector.detect(plate_img)
        raw_text = self.chars_to_text(chars)
        text = self.format_plate(raw_text)

        return text, chars

    def draw_vehicles(self, frame, vehicles):
        for (x1, y1, x2, y2, conf, label) in vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

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

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        vehicles = self.detect_vehicles(frame)
        plates = self.detect_plates(frame)

        results = []

        # vẽ vehicle
        self.draw_vehicles(frame, vehicles)

        #gom pts của các plate hợp lệ
        pts_list = []
        for plate in plates:
          #  bbox = plate["bbox"]          # (x1, y1, x2, y2)
            pts = plate["points"]         # keypoints hoặc None
            pts_list.append(np.array(pts, dtype=np.float32) if pts is not None else None)

        # làm mượt các pts không None
        pts_list_valid = [p for p in pts_list if p is not None]
        if pts_list_valid:
            pts_list_smooth = self.smooth_points(pts_list_valid)
        else:
            pts_list_smooth = []

        # gán lại pts_smooth vào plates
        idx_valid = 0
        for i, plate in enumerate(plates):
            if plate["points"] is not None:
                plates[i]["points_smooth"] = pts_list_smooth[idx_valid]
                idx_valid += 1
            else:
                plates[i]["points_smooth"] = None


        # xử lý từng plate
        for plate in plates:
            bbox = plate["bbox"]
            pts_smooth = plate.get("points_smooth", None)
            pts = plate["points"]  

            plate_img_for_char, plate_crop = self.prepare_plate_image(frame, bbox, pts)
            text, char_boxes = self.recognize_plate_text(plate_img_for_char)

            results.append({"bbox": bbox,"points": pts,"text": text,"chars": char_boxes})

            self.draw_plate_pose_and_text(frame,bbox,pts,text,plate_img_for_char,char_boxes=char_boxes,scale=2.0,offset=(10, -80)            )

        return frame, results


# ===== CHẠY ẢNH / VIDEO =====

import os

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

        ok = cv2.imwrite(out_path, out_img)
        if not ok:
            print("[-] Lưu ảnh thất bại tại:", out_path)

    if show:
        cv2.imshow("Result", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_img, results


def run_on_video(video_source, pipeline, save_path=None, show=True):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_source}")

    writer = None
    if save_path is not None:
        # Nếu save_path là folder -> auto tạo tên file theo video_source
        if os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
            base = os.path.basename(video_source)      # vd: input.mp4
            name, ext = os.path.splitext(base)         # input, .mp4
            if ext == "":
                ext = ".mp4"
            out_video_path = os.path.join(save_path, name + "_out6" + ext)
        else:
            # save_path là đường dẫn file đầy đủ
            folder = os.path.dirname(save_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            out_video_path = save_path

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise ValueError(f"Không tạo được VideoWriter: {out_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame, results = pipeline.process_frame(frame)

        if writer is not None:
            writer.write(out_frame)

        if show:
            cv2.imshow("ANPR Pipeline", out_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    return out_frame, results