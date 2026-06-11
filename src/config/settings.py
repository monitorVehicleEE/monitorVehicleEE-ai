# =========================================================
# GENERAL
# =========================================================

SHOW_WINDOW = False
SAVE_OUTPUT = True
STREAM_TIMING_LOG = False

# =========================================================
# STREAM
# =========================================================

STREAM_WIDTH = 640
JPEG_QUALITY = 75
START_READY_TIMEOUT = 60.0

# =========================================================
# CAMERA
# =========================================================

DROP_LATE_FRAMES = True
MAX_FRAME_SKIP = 1
VIDEO_BASE_DIR = "./dataset/vehicle/videos"

# =========================================================
# DETECTION STEP
# =========================================================

VEHICLE_STEP = 2
PLATE_STEP = 6
OCR_STEP = 8
VEHICLE_SAMPLE_STEP = 3

# =========================================================
# DETECTION LIMIT
# =========================================================

MAX_PLATE_FRAME_CHECKS = 2
MAX_OCR_READS = 3
MAX_CACHED_FRAMES = 30
MAX_PLATE_SAMPLE_RETRIES = 2

# =========================================================
# SHARPNESS
# =========================================================

PLATE_SHARPNESS_THRESHOLD = 150.0
CHAR_SHARPNESS_THRESHOLD = 80.0

# =========================================================
# TRACKING
# =========================================================

TRACK_MIN_LENGTH = 7
TRACK_MIN_VOTES = 3
TRACK_MAX_HISTORY = 20
TRACK_EXPIRE_FRAMES = 45
TRACK_LOST_APPEND_FRAMES = 3

# =========================================================
# MODEL PATH
# =========================================================

VEHICLE_MODEL_PATH = "./runs/detect/runs_vehicle/yolo11s_vehicle_v2/weights/best.pt"

PLATE_MODEL_PATH = "./runs/pose/runs_detect_plate/yl11s_dp_ver6/weights/best.pt"

OCR_MODEL_PATH = "./runs/detect/runs_read_plate/yolo11s_read_plate_v6/weights/best.pt"

# =========================================================
# YOLO
# =========================================================

DEVICE = 0
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
VEHICLE_CONF_THRESHOLD = 0.4
PLATE_CONF_THRESHOLD = 0.25
OCR_CONF_THRESHOLD = 0.3
PLATE_IMG_SIZE = 640
OCR_IMG_SIZE = 320

# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = "./dataset/output_test/"
EVENT_IMAGES_DIR = "./dataset/output_test/events"
CAMERA_API_URL = "http://localhost:8000"
AI_PUBLIC_URL = "http://localhost:8001"
EVENT_POST_QUEUE_LIMIT = 50
ACTIVE_EVENT_SEND_INTERVAL_FRAMES = 10
ACTIVE_EVENT_PLATE_CONFIDENCE = 0.75
ACTIVE_EVENT_NO_PLATE_MIN_FRAMES = 30
