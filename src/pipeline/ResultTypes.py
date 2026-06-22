from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class VehicleResult:
    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray
    detections: list


@dataclass
class PlateResult:
    camera_id: str
    frame_id: int
    source_frame_id: int
    track_id: int
    track_started_at: int
    success: bool
    reason: str
    vehicle_bbox: Tuple[int, int, int, int]
    vehicle_label: str
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    plate_points: Optional[np.ndarray] = None
    plate_img: Optional[np.ndarray] = None
    sharpness: float = 0.0


@dataclass
class OCRResult(PlateResult):
    text: str = ""
    avg_confidence: float = 0.0
    char_count: int = 0
