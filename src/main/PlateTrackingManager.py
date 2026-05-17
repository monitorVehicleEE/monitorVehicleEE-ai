from collections import Counter
from datetime import datetime
import re
from PlateStatus import PlateStatus


class PlateTrackingManager:
    def __init__(self, min_length=7, min_votes=3, max_history=20, 
                 expire_frames=60, min_confidence=0.6):
        """
        Parameters:
        -----------
        min_length : int (default=7)
            Độ dài tối thiểu của biển số hợp lệ (VN: 7-9 ký tự)
            
        min_votes : int (default=3)
            Số lần xuất hiện tối thiểu để xác nhận plate text ổn định
            Voting giúp lọc lỗi OCR tạm thời
            
        max_history : int (default=20)
            Số lượng candidates tối đa lưu trong memory
            Giá trị cao -> memory usage tăng nhưng voting chính xác hơn
            
        expire_frames : int (default=60)
            Số frame tối đa một track có thể mất dấu trước khi bị xóa
            60 frames ≈ 2-3 giây @ 25fps
            
        min_confidence : float (default=0.6)
            Ngưỡng confidence tối thiểu để coi là READABLE
            < 0.6 → LOW_CONFIDENCE
        """
        self.min_length = min_length
        self.min_votes = min_votes
        self.max_history = max_history
        self.expire_frames = expire_frames
        self.min_confidence = min_confidence
        self.memory = {}           # Active tracks
        self.finalized = {}        # Finalized tracks


    def init_track(self, track_id, vehicle_type="unknown"):
        self.memory[track_id] = {
            "track_id": track_id,
            "vehicle_type": vehicle_type,           
            # Trạng thái
            "has_plate": False,
            "plate_detected": False,
            "plate_readable": False,          
            # Voting
            "candidates": [],
            "best_text": "",
            "best_count": 0,       
            # Score
            "best_score": 0.0,            
            # Frame tracking
            "first_seen_frame": 0,
            "last_seen_frame": 0,           
            # Timestamp
            "first_seen_time": datetime.now().isoformat(),
            "last_seen_time": datetime.now().isoformat(),           
            # Statistics
            "successful_reads": 0,            
            # Bounding box
            "last_bbox": None,          
            # Finalized flag
            "finalized": False,
            # Status & metrics
            "plate_status": PlateStatus.NO_PLATE.value,
            "detection_attempts": 0,  # Số lần detect được plate bbox
            "read_attempts": 0,       # Số lần thử đọc ký tự
            "avg_sharpness": 0.0,
            "avg_confidence": 0.0,
            "no_plate_frames": 0,
            "blur_frames": 0,
        }
    
    def update_no_plate(self, track_id, frame_index):
        """
        Gọi khi vehicle detected nhưng không có plate bbox

        """
        if track_id not in self.memory:
            self.init_track(track_id)
        
        mem                     = self.memory[track_id]
        mem["no_plate_frames"] += 1
        mem["last_seen_frame"]  = frame_index
        mem["last_seen_time"]   = datetime.now().isoformat()
        
        # Xác nhận NO_PLATE sau 15 frames liên tục
        if mem["no_plate_frames"] > 15:
            mem["plate_status"] = PlateStatus.NO_PLATE.value
    
    def update_unreadable_plate(self, track_id, frame_index, sharpness, reason="blur"):
        """
        Gọi khi detect được plate nhưng không đọc được ký tự
        """
        if track_id not in self.memory:
            self.init_track(track_id)
        
        mem = self.memory[track_id]
        mem["has_plate"]            = True
        mem["plate_detected"]       = True
        mem["detection_attempts"]   += 1
        mem["read_attempts"]        += 1
        mem["blur_frames"]          += 1
        mem["last_seen_frame"]      = frame_index
        mem["last_seen_time"]       = datetime.now().isoformat()
        
        # Cập nhật sharpness trung bình (tránh chia 0)
        n = mem["read_attempts"]
        if n > 0:
            mem["avg_sharpness"] = ((mem["avg_sharpness"] * (n-1)) + sharpness) / n
        
        # Xác nhận UNREADABLE nếu 5 frame liên tục mờ và chưa đọc được lần nào
        if mem["blur_frames"] > 5 and mem["successful_reads"] == 0:
            mem["plate_status"] = PlateStatus.UNREADABLE.value


    # === Validation ===
    def clean_text(self, text):
        """Loại bỏ ký tự format khỏi plate text"""
        if text is None:
            return ""
        text = text.upper()
        text = text.replace("-", "").replace(".", "").replace(" ", "")
        return text


    def is_valid_plate(self, text):
        """
        Kiểm tra plate text có hợp lệ không
        """
        clean = self.clean_text(text)
        
        if len(clean) < self.min_length:
            return False
        
        has_digit = any(c.isdigit() for c in clean)
        has_alpha = any(c.isalpha() for c in clean)
        
        return has_digit and has_alpha


    def compute_score(self, text, sharpness=0, char_count=0, avg_char_conf=0):
        """
        Tính điểm chất lượng cho một lần đọc plate
        """
        clean = self.clean_text(text)
        score = 0.0
        
        # Độ dài (max 10 points)
        score += min(len(clean), 10) * 1
        
        # Sharpness (max 40 points)
        score += min(sharpness / 10.0, 40)
        
        # Char count (max 20 points)
        score += min(char_count * 2, 20)
        
        # Confidence (max 30 points)
        score += avg_char_conf * 30
        
        return score
    
    def update_plate(self, track_id, text, frame_index=0, sharpness=0,
                     char_count=0, avg_char_conf=0, bbox=None):
        """
        Cập nhật khi đọc được plate text
        """
        if track_id not in self.memory:
            self.init_track(track_id)
        
        mem                         = self.memory[track_id]
        mem["has_plate"]            = True
        mem["plate_detected"]       = True
        mem["detection_attempts"]   += 1 
        mem["read_attempts"]        += 1
        mem["last_seen_frame"]      = frame_index
        mem["last_seen_time"]       = datetime.now().isoformat()
        
        if bbox is not None:
            mem["last_bbox"] = bbox
        
        # Cập nhật sharpness & confidence trung bình
        n = mem["read_attempts"]
        mem["avg_sharpness"] = ((mem["avg_sharpness"] * (n-1)) + sharpness) / n
        mem["avg_confidence"] = ((mem["avg_confidence"] * (n-1)) + avg_char_conf) / n
        
        # Validate plate
        if not self.is_valid_plate(text):
            mem["plate_status"] = PlateStatus.UNREADABLE.value
            return
        
        mem["plate_readable"] = True
        mem["successful_reads"] += 1
        mem["candidates"].append(text)
        
        # Limit history
        if len(mem["candidates"]) > self.max_history:
            mem["candidates"] = mem["candidates"][-self.max_history:]
        
        # Voting: tìm plate text xuất hiện nhiều nhất
        counter = Counter(mem["candidates"])
        best_text, best_count = counter.most_common(1)[0]
        mem["best_text"] = best_text
        mem["best_count"] = best_count
        
        # Xác định status
        if avg_char_conf < self.min_confidence:
            mem["plate_status"] = PlateStatus.LOW_CONFIDENCE.value
        elif best_count >= self.min_votes:
            mem["plate_status"] = PlateStatus.VERIFIED.value
        else:
            mem["plate_status"] = PlateStatus.READABLE.value
        
        # Score
        score = self.compute_score(text, sharpness, char_count, avg_char_conf)
        if score > mem["best_score"]:
            mem["best_score"] = score


    def get_stable_text(self, track_id):
        """
        Lấy plate text đã được voting ổn định
        
        Returns:
        --------
        str : Plate text nếu đã đủ votes, ngược lại ""
        """
        if track_id not in self.memory:
            return ""
        
        mem = self.memory[track_id]
        
        if mem["best_count"] < self.min_votes:
            return ""
        
        return mem["best_text"]


    def get_track_result(self, track_id):
        """
        Lấy kết quả đầy đủ của một track
        """
        if track_id not in self.memory:
            return None
        
        mem = self.memory[track_id]
        stable_text = self.get_stable_text(track_id)
        
        return {
            "track_id"          : track_id,
            "vehicle_type"      : mem["vehicle_type"],
            "plate_text"        : stable_text,
            "vote_count"        : mem["best_count"],
            "best_score"        : mem["best_score"],
            "read_attempts"     : mem["read_attempts"],
            "successful_reads"  : mem["successful_reads"],
            "bbox"              : mem["last_bbox"],
            "first_seen"        : mem["first_seen_time"],
            "last_seen"         : mem["last_seen_time"],
            "plate_detected"    : mem["plate_detected"],
            "plate_readable"    : mem["plate_readable"],
            "status"            : mem["plate_status"],
            "avg_confidence"    : mem["avg_confidence"],
            "avg_sharpness"     : mem["avg_sharpness"]
        }


    def remove_expired_tracks(self, current_frame):
        """
        Xóa các track đã mất dấu quá lâu
        track trước khi xóa để đảm bảo data không mất
        """
        remove_ids = []
        
        for track_id, mem in self.memory.items():
            last_seen = mem["last_seen_frame"]
            
            # Kiểm tra track đã expire chưa
            if current_frame - last_seen > self.expire_frames:
                # Finalize trước khi xóa
                if not mem.get("finalized", False):
                    self.finalize_track(track_id)
                
                remove_ids.append(track_id)
        
        # Xóa khỏi active memory
        for tid in remove_ids:
            del self.memory[tid]
        
        if remove_ids:
            print(f"[INFO] Removed {len(remove_ids)} expired tracks at frame {current_frame}")
    
    
    def finalize_track(self, track_id):
        """
        Lưu track vào finalized storage trước khi xóa khỏi memory
        track_id : int
            ID của track cần finalize
        """
        if track_id not in self.memory:
            return
        
        mem = self.memory[track_id]
        mem["finalized"] = True
        mem["finalized_time"] = datetime.now().isoformat()
        
        # Copy sang finalized storage
        self.finalized[track_id] = mem.copy()
        
        print(f"[INFO] Finalized track {track_id}: "
              f"status={mem['plate_status']}, "
              f"text={mem.get('best_text', 'N/A')}")


    def finalize_all_active_tracks(self):
        """
        Finalize tất cả active tracks (gọi khi kết thúc video)
        """
        for track_id in list(self.memory.keys()):
            if not self.memory[track_id].get("finalized", False):
                self.finalize_track(track_id)
        
        print(f"[INFO] Finalized all {len(self.memory)} active tracks")


    def _build_result_from_mem(self, mem):
        """
        Build result dict từ memory object
        """
        stable_text = (mem.get("best_text", "") 
                      if mem.get("best_count", 0) >= self.min_votes 
                      else "")
        
        return {
            "track_id": mem["track_id"],
            "vehicle_type": mem["vehicle_type"],
            "plate_text": stable_text,
            "vote_count": mem.get("best_count", 0),
            "best_score": mem.get("best_score", 0.0),
            "read_attempts": mem.get("read_attempts", 0),
            "successful_reads": mem.get("successful_reads", 0),
            "bbox": mem.get("last_bbox"),
            "first_seen": mem.get("first_seen_time"),
            "last_seen": mem.get("last_seen_time"),
            "plate_detected": mem.get("plate_detected", False),
            "plate_readable": mem.get("plate_readable", False),
            "status": mem.get("plate_status", PlateStatus.NO_PLATE.value),
            "avg_confidence": mem.get("avg_confidence", 0.0),
            "avg_sharpness": mem.get("avg_sharpness", 0.0),
            "finalized": mem.get("finalized", False)
        }


    def export_all_results(self):
        """
        Export TẤT CẢ kết quả (cả active và finalized)
        Returns:: Danh sách tất cả tracks đã xử lý
        """
        results = []
        
        # Export active tracks
        for track_id in self.memory:
            result = self.get_track_result(track_id)
            if result:
                results.append(result)
        
        # Export finalized tracks (tránh duplicate)
        for track_id, mem in self.finalized.items():
            if track_id not in self.memory:  # Chỉ thêm nếu không còn trong active
                result = self._build_result_from_mem(mem)
                if result:
                    results.append(result)
        
        return results
    
    
    def clear_finalized(self):
        """
        Xóa finalized storage sau khi đã export xong
        Gọi sau khi lưu JSON thành công
        """
        count = len(self.finalized)
        self.finalized.clear()
        
        if count > 0:
            print(f"[INFO] Cleared {count} finalized tracks")