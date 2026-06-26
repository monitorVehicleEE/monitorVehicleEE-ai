import os


def _get_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_device(default):
    value = os.getenv("DEVICE")
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return value


# =========================================================
# GENERAL
# =========================================================

SHOW_WINDOW = _get_bool("SHOW_WINDOW", False)
SAVE_OUTPUT = _get_bool("SAVE_OUTPUT", True)
STREAM_TIMING_LOG = _get_bool("STREAM_TIMING_LOG", False)

# =========================================================
# STREAM
# =========================================================

STREAM_WIDTH = _get_int("STREAM_WIDTH", 640)
JPEG_QUALITY = _get_int("JPEG_QUALITY", 75)
START_READY_TIMEOUT = _get_float("START_READY_TIMEOUT", 60.0)

# =========================================================
# CAMERA
# =========================================================

DROP_LATE_FRAMES = _get_bool("DROP_LATE_FRAMES", True)
MAX_FRAME_SKIP = _get_int("MAX_FRAME_SKIP", 1)
VIDEO_BASE_DIR = os.getenv("VIDEO_BASE_DIR", "./dataset/vehicle/videos")

# =========================================================
# DETECTION STEP
# =========================================================

VEHICLE_STEP = _get_int("VEHICLE_STEP", 2)
PLATE_STEP = _get_int("PLATE_STEP", 3)
OCR_STEP = _get_int("OCR_STEP", 4)
VEHICLE_SAMPLE_STEP = _get_int("VEHICLE_SAMPLE_STEP", 3)

# =========================================================
# DETECTION LIMIT
# =========================================================

MAX_PLATE_FRAME_CHECKS = _get_int("MAX_PLATE_FRAME_CHECKS", 4)
MAX_OCR_READS = _get_int("MAX_OCR_READS", 3)
MAX_CACHED_FRAMES = _get_int("MAX_CACHED_FRAMES", 30)
MAX_PLATE_SAMPLE_RETRIES = _get_int("MAX_PLATE_SAMPLE_RETRIES", 2)

# =========================================================
# SHARPNESS
# =========================================================

PLATE_SHARPNESS_THRESHOLD = _get_float("PLATE_SHARPNESS_THRESHOLD", 150.0)
CHAR_SHARPNESS_THRESHOLD = _get_float("CHAR_SHARPNESS_THRESHOLD", 80.0)

# =========================================================
# TRACKING
# =========================================================

TRACK_MIN_LENGTH = _get_int("TRACK_MIN_LENGTH", 7)
TRACK_MIN_VOTES = _get_int("TRACK_MIN_VOTES", 3)
TRACK_MAX_HISTORY = _get_int("TRACK_MAX_HISTORY", 20)
TRACK_EXPIRE_FRAMES = _get_int("TRACK_EXPIRE_FRAMES", 45)
TRACK_LOST_APPEND_FRAMES = _get_int("TRACK_LOST_APPEND_FRAMES", 3)

# =========================================================
# MODEL PATH
# =========================================================

VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "./model/pytorch/vehicle/best.pt")

PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "./model/pytorch/plate/best.pt")

OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH", "./model/pytorch/char/best.pt")

# =========================================================
# YOLO
# =========================================================

DEVICE = _get_device(0)
IMG_SIZE = _get_int("IMG_SIZE", 640)
CONF_THRESHOLD = _get_float("CONF_THRESHOLD", 0.25)
VEHICLE_CONF_THRESHOLD = _get_float("VEHICLE_CONF_THRESHOLD", 0.4)
PLATE_CONF_THRESHOLD = _get_float("PLATE_CONF_THRESHOLD", 0.25)
OCR_CONF_THRESHOLD = _get_float("OCR_CONF_THRESHOLD", 0.3)
PLATE_IMG_SIZE = _get_int("PLATE_IMG_SIZE", 640)
OCR_IMG_SIZE = _get_int("OCR_IMG_SIZE", 320)

# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./dataset/output_test/")
EVENT_IMAGES_DIR = os.getenv("EVENT_IMAGES_DIR", "./dataset/output_test/events")
CAMERA_API_URL = os.getenv("CAMERA_API_URL", "http://localhost:8000")
AI_PUBLIC_URL = os.getenv("AI_PUBLIC_URL", "http://localhost:8001")
EVENT_POST_QUEUE_LIMIT = _get_int("EVENT_POST_QUEUE_LIMIT", 50)
ACTIVE_EVENT_SEND_INTERVAL_FRAMES = _get_int("ACTIVE_EVENT_SEND_INTERVAL_FRAMES", 5)
ACTIVE_EVENT_PLATE_CONFIDENCE = _get_float("ACTIVE_EVENT_PLATE_CONFIDENCE", 0.7)
ACTIVE_EVENT_NO_PLATE_MIN_FRAMES = _get_int("ACTIVE_EVENT_NO_PLATE_MIN_FRAMES", 30)
