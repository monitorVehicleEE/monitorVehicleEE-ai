from collections import Counter
from datetime import datetime
import re
from src.main.PlateStatus import PlateStatus


class TrackingManager:
    def __init__(self, min_length=7, max_length=9, min_votes=2, max_history=20,
                 expire_frames=60, min_confidence=0.6,
                 no_plate_ratio=0.9, min_sample_frames=10, blur_ratio=0.7):
        """
        expire_frames : 
            Số frame tối đa một track có thể mất dấu trước khi bị xóa
            60 frames ≈ 2-3 giây @ 25fps
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_votes = min_votes
        self.max_history = max_history
        self.expire_frames = expire_frames
        self.min_confidence = min_confidence
        self.memory = {}            # Active tracks
        self.finalized = {}         # Finalized tracks
        self.no_plate_ratio = no_plate_ratio
        self.min_sample_frames = min_sample_frames
        self.blur_ratio = blur_ratio

    def init_track(self, track_id, vehicle_type="unknown", frame_index=0):
        self.memory[track_id] = {
            "track_id": track_id,
            "vehicle_type": vehicle_type,
            "vehicle_type_votes": {},
            "vehicle_type_scores": {},

            # Trạng thái
            "has_plate": False,
            "plate_detected": False,
            "plate_readable": False,

            # Voting
            "candidates": [],   # list[{ "text": ..., "score": ... }]
            "best_text": "",
            "best_count": 0,
            "best_score": 0.0,

            # Frame tracking
            "first_seen_frame": frame_index,
            "last_seen_frame": frame_index,

            # Timestamp
            "first_seen_time": datetime.now().isoformat(),
            "last_seen_time": datetime.now().isoformat(),
            "missing_frames": 0,

            # Statistics
            "successful_reads": 0,
            "read_attempts": 0,
            "detection_attempts": 0,

            # Bounding box
            "last_bbox_plate": None,

            # Finalized flag
            "finalized": False,

            # Status & metrics
            "plate_status": PlateStatus.NO_PLATE.value,
            "avg_sharpness": 0.0,
            "avg_confidence": 0.0,
            "no_plate_frames": 0,
            "blur_frames": 0,

            # Thông tin về ANPR adaptive
            "last_anpr_frame": -999,
            "adaptive_anpr_step": 6,   # default: mỗi ~6 frame thử ANPR 1 lần

            # chất lượng tốt nhất từng thấy (để so sánh)
            "best_quality": 0.0,

            # burst mode (nếu phát hiện frame rất tốt thì tạm thời tăng tần suất)
            "burst_mode": False,
            "burst_until": 0,

            # Top-N plate frames để dùng / phân tích thêm
            # Vehicle frames (top-N)
            "best_vehicle_frames": [],
            "best_plate_frames": [],
            "vehicle_quality": 0.0,

            "display_text": "",
            "vehicle_bbox": None, 
            "last_plate_view": None,
        }

    def update_vehicle_type(self, track_id, vehicle_type, confidence=0.0):
        if track_id not in self.memory:
            self.init_track(track_id, vehicle_type=vehicle_type)

        mem = self.memory[track_id]
        label = vehicle_type or "unknown"
        votes = mem.setdefault("vehicle_type_votes", {})
        scores = mem.setdefault("vehicle_type_scores", {})

        votes[label] = votes.get(label, 0) + 1
        scores[label] = scores.get(label, 0.0) + float(confidence or 0.0)

        best_label = max(
            votes,
            key=lambda item: (votes[item], scores.get(item, 0.0)),
        )
        mem["vehicle_type"] = best_label
        mem["vehicle_confidence"] = (
            scores.get(best_label, 0.0) / max(1, votes.get(best_label, 0))
        )
        return best_label

    # # ====== ADAPTIVE ANPR SUPPORT ======
    def can_run_anpr(self, track_id, frame_index):
        """
        Quyết định có nên chạy OCR tại frame này cho track này hay không,
        dựa vào khoảng cách frame và chế độ burst_mode.
        """
        if track_id not in self.memory:
            return True

        mem = self.memory[track_id]
        step = mem.get("adaptive_anpr_step", 6)

        # Nếu đang ở burst_mode (ví dụ vừa thấy frame rất đẹp) -> tăng tần suất
        if mem.get("burst_mode", False) and frame_index <= mem.get("burst_until", 0):
            step = 2  # đọc dày hơn tạm thời

        return (frame_index - mem.get("last_anpr_frame", -999)) >= step

    def update_after_anpr(self, track_id, frame_index, quality_score):
        """
        Gọi sau mỗi lần ANPR (kể cả thành công hay không) để cập nhật logic adaptive.
        quality_score: có thể dùng sharpness + avg_conf...
        """
        if track_id not in self.memory:
            return

        mem = self.memory[track_id]
        mem["last_anpr_frame"] = frame_index

        # Cập nhật best_quality
        if quality_score > mem.get("best_quality", 0.0):
            mem["best_quality"] = quality_score
            # Nếu frame này rất tốt (cao hơn ngưỡng), bật burst mode 1 đoạn ngắn
            mem["burst_mode"] = True
            mem["burst_until"] = frame_index + 10  # trong 10 frame tới đọc dày hơn
        else:
            # Nếu đã qua giai đoạn burst
            if frame_index > mem.get("burst_until", 0):
                mem["burst_mode"] = False

    def update_best_vehicle_frame(self, track_id, frame_index, bbox, sharpness,
                                  priority, max_samples=5, vehicle_img=None,
                                  frame_ref=None):
        if track_id not in self.memory:
            return

        mem = self.memory[track_id]

        sample = {
            "track_id": track_id,
            "frame_idx": frame_index,
            "bbox": bbox,
            "sharpness": sharpness,
            "priority": priority,
            "vehicle_img": vehicle_img,
            "frame_ref": frame_ref if frame_ref is not None else frame_index,
            "plate_checked": False
        }

        mem["best_vehicle_frames"].append(sample)

        mem["best_vehicle_frames"] = sorted(
            mem["best_vehicle_frames"],
            key=lambda x: x["priority"],
            reverse=True
        )[:max_samples]

        best_v = mem["best_vehicle_frames"][0]
        mem["vehicle_quality"] = best_v["priority"]
    
    def get_best_vehicle_samples(self, track_id, top_k=2):
        if track_id not in self.memory:
            return []

        mem = self.memory[track_id]

        return mem.get("best_vehicle_frames", [])[:top_k]

    def update_best_plate_frame(self, track_id, frame_index, bbox, sharpness, priority,
                                plate_img=None, pts=None, max_samples=5 , vehicle_bbox = None):
        """
        Lưu lại top-N frame plate tốt nhất cho mỗi track.
        priority: điểm ưu tiên (ví dụ: sharpness + 0.01*area)
        plate_img: ảnh biển đã warp/crop (dùng cho OCR worker)
        pts: keypoints pose (nếu cần vẽ lại)
        """
        if track_id not in self.memory:
            return

        mem = self.memory[track_id]

        for old in mem["best_plate_frames"]:
            if frame_index == old["frame_idx"]:
                if priority <= old.get("priority", 0):
                    return
                mem["best_plate_frames"].remove(old)
                break
        sample = {
            "frame_idx": frame_index,
            "bbox": bbox,
            "sharpness": sharpness,
            "priority": priority,
            "plate_img": plate_img,
            "pts": pts,
            "vehicle_bbox": vehicle_bbox
        }
        mem["best_plate_frames"].append(sample)

        mem["best_plate_frames"] = sorted(
            mem["best_plate_frames"],
            key=lambda x: x["priority"],
            reverse=True
        )[:max_samples]


    def get_best_plate_samples(self, track_id, top_k=2):
        if track_id not in self.memory:
            return []
        mem = self.memory[track_id]
        return mem.get("best_plate_frames", [])[:top_k]

    # ====== STATUS & PLATE UPDATE ======
    def update_plate_status(self, track_id):
        mem = self.memory[track_id]

        total_frames = mem["last_seen_frame"] - mem["first_seen_frame"] + 1
        read_attempts = mem.get("read_attempts", 0)
        successful_reads = mem.get("successful_reads", 0)

        # NO_PLATE
        if not mem["plate_detected"]:
            if total_frames >= self.min_sample_frames:
                no_plate_rate = mem["no_plate_frames"] / max(total_frames, 1)
                if no_plate_rate >= self.no_plate_ratio:
                    mem["plate_status"] = PlateStatus.NO_PLATE.value
            return

        # UNREADABLE
        if successful_reads == 0 and read_attempts > 0:
            if read_attempts >= self.min_sample_frames:
                actual_blur_ratio = mem["blur_frames"] / max(read_attempts, 1)
                if actual_blur_ratio >= self.blur_ratio:
                    mem["plate_status"] = PlateStatus.UNREADABLE.value
                    return
            mem["plate_status"] = PlateStatus.READABLE.value
            return

        # VERIFIED / LOW_CONFIDENCE / READABLE
        if successful_reads > 0:
            best_count = mem.get("best_count", 0)
            avg_conf = mem.get("avg_confidence", 0)

            if best_count >= self.min_votes and avg_conf >= self.min_confidence:
                mem["plate_status"] = PlateStatus.VERIFIED.value
            elif avg_conf < self.min_confidence:
                mem["plate_status"] = PlateStatus.LOW_CONFIDENCE.value
            else:
                mem["plate_status"] = PlateStatus.READABLE.value

    def update_no_plate(self, track_id, frame_index):
        if track_id not in self.memory:
            self.init_track(track_id, frame_index=frame_index)

        mem = self.memory[track_id]
        mem["no_plate_frames"] += 1
        mem["last_seen_frame"] = frame_index
        mem["last_seen_time"] = datetime.now().isoformat()

    def update_unreadable_plate(self, track_id, frame_index, sharpness, reason="blur"):
        if track_id not in self.memory:
            self.init_track(track_id, frame_index=frame_index)

        mem = self.memory[track_id]
        mem["has_plate"] = True
        mem["plate_detected"] = True
        mem["detection_attempts"] += 1
        mem["read_attempts"] += 1
        mem["blur_frames"] += 1
        mem["last_seen_frame"] = frame_index
        mem["last_seen_time"] = datetime.now().isoformat()
        mem["no_plate_frames"] = 0

        n = mem["read_attempts"]
        if n > 0:
            mem["avg_sharpness"] = ((mem["avg_sharpness"] * (n - 1)) + sharpness) / n

        self.update_plate_status(track_id)

    # ====== VALIDATION ======
    def clean_text(self, text):
        if text is None:
            return ""
        text = text.upper()
        text = text.replace("-", "").replace(".", "").replace(" ", "")
        return text

    def is_valid_plate(self, text):
        clean = self.clean_text(text)
        if len(clean) < self.min_length or len(clean) > self.max_length:
            return False
        has_digit = any(c.isdigit() for c in clean)
        has_alpha = any(c.isalpha() for c in clean)
        return has_digit and has_alpha

    def compute_score(self, text, sharpness=0, char_count=0, avg_conf=0):
        clean = self.clean_text(text)
        score = 0.0
        score += min(len(clean), 10) * 1
        score += min(sharpness / 10.0, 40)
        score += min(char_count * 2, 20)
        score += avg_conf * 30
        return score

    def update_plate(self, track_id, text, frame_index=0, sharpness=0,
                     char_count=0, avg_conf=0, bbox=None):
        if track_id not in self.memory:
            self.init_track(track_id, frame_index=frame_index)

        mem = self.memory[track_id]
        mem["has_plate"] = True
        mem["plate_detected"] = True
        mem["detection_attempts"] += 1
        mem["read_attempts"] += 1
        mem["last_seen_frame"] = frame_index
        mem["last_seen_time"] = datetime.now().isoformat()
        mem["no_plate_frames"] = 0

        if bbox is not None:
            mem["last_bbox_plate"] = bbox

        n = mem["read_attempts"]
        mem["avg_sharpness"] = ((mem["avg_sharpness"] * (n - 1)) + sharpness) / n
        mem["avg_confidence"] = ((mem["avg_confidence"] * (n - 1)) + avg_conf) / n

        if not self.is_valid_plate(text):
            self.update_plate_status(track_id)
            return

        score = self.compute_score(text, sharpness, char_count, avg_conf)

        mem["plate_readable"] = True
        mem["successful_reads"] += 1

        mem["candidates"].append({
            "text": text,
            "score": score
        })

        top_k = 5
        mem["candidates"] = sorted(
            mem["candidates"],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        texts = [c["text"] for c in mem["candidates"]]
        if texts:
            counter = Counter(texts)
            best_text, best_count   = counter.most_common(1)[0]
            mem["best_text"]        = best_text
            mem["best_count"]       = best_count
            mem["best_score"]       = mem["candidates"][0]["score"]
        else:
            mem["best_text"]    = ""
            mem["best_count"]   = 0

        self.update_plate_status(track_id)

    def get_stable_text(self, track_id):
        if track_id not in self.memory:
            return ""
        mem = self.memory[track_id]
        if mem["successful_reads"] == 0:
            return ""
        if mem.get("best_text") and mem.get("best_count", 0) >= self.min_votes:
            return mem["best_text"]
        return mem.get("best_text", "")

    def get_track_result(self, track_id):
        if track_id not in self.memory:
            return None
        return self._build_result_from_mem(self.memory[track_id])

    # ====== LIFECYCLE ======
    def remove_expired_tracks(self, current_frame):
        remove_ids = []
        finalized_results = []
        for track_id, mem in self.memory.items():
            last_seen = mem["last_seen_frame"]
            if current_frame - last_seen > self.expire_frames:
                if not mem.get("finalized", False):
                    result = self._build_result_from_mem(mem)
                    if result:
                        finalized_results.append(result)
                    self.finalize_track(track_id)
                remove_ids.append(track_id)

        for tid in remove_ids:
            del self.memory[tid]

        return finalized_results

    def finalize_track(self, track_id):
        if track_id not in self.memory:
            return
        mem = self.memory[track_id]
        mem["finalized"] = True
        mem["finalized_time"] = datetime.now().isoformat()
        self.finalized[track_id] = mem.copy()

    def finalize_all_active_tracks(self):
        for track_id in list(self.memory.keys()):
            if not self.memory[track_id].get("finalized", False):
                self.finalize_track(track_id)

    def _select_best_plate_sample(self, mem):
        samples = [
            sample for sample in mem.get("best_plate_frames", [])
            if sample.get("plate_img") is not None
        ]
        if not samples:
            return {}

        stable_text = mem.get("best_text") or ""
        readable_samples = [
            sample for sample in samples
            if sample.get("ocr_success")
        ]

        if stable_text:
            stable_samples = [
                sample for sample in readable_samples
                if sample.get("ocr_text") == stable_text
            ]
            if stable_samples:
                return max(
                    stable_samples,
                    key=lambda sample: (
                        sample.get("ocr_score", 0),
                        sample.get("priority", 0),
                    )
                )

        if readable_samples:
            return max(
                readable_samples,
                key=lambda sample: (
                    sample.get("ocr_score", 0),
                    sample.get("priority", 0),
                )
            )

        usable_samples = [
            sample for sample in samples
            if not sample.get("ocr_failed")
        ]
        if usable_samples:
            return max(
                usable_samples,
                key=lambda sample: sample.get("priority", 0)
            )

        return max(samples, key=lambda sample: sample.get("priority", 0))

    def _select_best_vehicle_sample(self, mem, best_plate):
        samples = [
            sample for sample in mem.get("best_vehicle_frames", [])
            if sample.get("vehicle_img") is not None
        ]
        if not samples:
            return {}

        if best_plate:
            plate_frame_idx = best_plate.get("frame_idx")
            plate_vehicle_bbox = best_plate.get("vehicle_bbox")

            matching_samples = [
                sample for sample in samples
                if (
                    sample.get("frame_idx") == plate_frame_idx
                    or sample.get("bbox") == plate_vehicle_bbox
                )
            ]
            if matching_samples:
                return max(
                    matching_samples,
                    key=lambda sample: sample.get("priority", 0)
                )

        return max(samples, key=lambda sample: sample.get("priority", 0))

    def _build_result_from_mem(self, mem):
        stable_text = (mem.get("best_text", "")
                       if mem.get("successful_reads", 0) > 0
                       else "")
        best_plate = self._select_best_plate_sample(mem)
        best_vehicle = self._select_best_vehicle_sample(mem, best_plate)
        plate_bbox = best_plate.get("bbox") or mem.get("last_bbox_plate")
        vehicle_bbox = best_vehicle.get("bbox") or mem.get("vehicle_bbox")

        return {
            "track_id": mem["track_id"],
            "vehicle_type": mem["vehicle_type"],
            "vehicle_type_votes": dict(mem.get("vehicle_type_votes", {})),
            "plate_text": stable_text,
            "vote_count": mem.get("best_count", 0),
            "best_score": mem.get("best_score", 0.0),
            "read_attempts": mem.get("read_attempts", 0),
            "detection_attempts": mem.get("detection_attempts", 0),
            "successful_reads": mem.get("successful_reads", 0),
            "bbox": plate_bbox,
            "bbox_plate": plate_bbox,
            "vehicle_bbox": vehicle_bbox,
            "last_vehicle_bbox": mem.get("vehicle_bbox"),
            "best_vehicle_frame": best_vehicle.get("frame_idx"),
            "best_plate_frame": best_plate.get("frame_idx"),
            "vehicle_confidence": mem.get("vehicle_confidence"),
            "first_seen_frame": mem.get("first_seen_frame"),
            "last_seen_frame": mem.get("last_seen_frame"),
            "first_seen": mem.get("first_seen_time"),
            "last_seen": mem.get("last_seen_time"),
            "plate_detected": mem.get("plate_detected", False),
            "plate_readable": mem.get("plate_readable", False),
            "status": mem.get("plate_status", PlateStatus.NO_PLATE.value),
            "avg_confidence": mem.get("avg_confidence", 0.0),
            "avg_sharpness": mem.get("avg_sharpness", 0.0),
            "finalized": mem.get("finalized", False),
            "best_vehicle_img": best_vehicle.get("vehicle_img"),
            "best_plate_img": best_plate.get("plate_img"),
        }

    def export_all_results(self):
        results = []
        for track_id in self.memory:
            result = self.get_track_result(track_id)
            if result:
                results.append(result)
        for track_id, mem in self.finalized.items():
            if track_id not in self.memory:
                result = self._build_result_from_mem(mem)
                if result:
                    results.append(result)
        return results

    def clear_finalized(self):
        count = len(self.finalized)
        self.finalized.clear()
        if count > 0:
            print(f"[INFO] Cleared {count} finalized tracks")

    # ====== DISPLAY SUPPORT ======
    def update_display_info(self, track_id, text):
        if track_id not in self.memory:
            return
        mem = self.memory[track_id]
        mem["display_text"] = text

    def get_display_info(self, track_id):
        if track_id not in self.memory:
            return None
        mem = self.memory[track_id]
        text = mem.get("display_text", None)
        if not text:
            return None
        return {
            "track_id": track_id,
            "text": text
        }
