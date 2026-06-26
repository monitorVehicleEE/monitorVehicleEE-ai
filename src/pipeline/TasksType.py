from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class VehicleTask:
    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class PlateTask:
    camera_id: str
    frame_id: int
    source_frame_id: int
    track_id: int
    # dùng để chống kết quả cũ trả về muộn.
    track_started_at: int
    vehicle_bbox: Tuple[int, int, int, int]
    vehicle_label: str
    frame: np.ndarray


@dataclass
class CharTask:
    camera_id: str
    frame_id: int
    source_frame_id: int
    track_id: int
    track_started_at: int
    vehicle_bbox: Tuple[int, int, int, int]
    vehicle_label: str
    plate_bbox: Tuple[int, int, int, int]
    plate_points: Optional[np.ndarray]
    plate_img: np.ndarray
    sharpness: float
