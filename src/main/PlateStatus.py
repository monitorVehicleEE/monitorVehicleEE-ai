from enum import Enum

class PlateStatus(Enum):
    NO_PLATE = "no_plate"           # Không detect được biển
    UNREADABLE = "unreadable"       # Detect được nhưng không đọc được
    LOW_CONFIDENCE = "low_confidence" # Đọc được nhưng confidence thấp
    READABLE = "readable"           # Đọc được và tin cậy
    VERIFIED = "verified"           # Đã voting ổn định