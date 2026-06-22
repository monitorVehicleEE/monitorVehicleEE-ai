from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    camera_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray
