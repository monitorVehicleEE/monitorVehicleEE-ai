import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AUTO_APPROVE_PLATE_CONFIDENCE = float(
    os.getenv("AUTO_APPROVE_PLATE_CONFIDENCE", "0.75")
)


def normalize_plate(plate):
    if not plate:
        return None

    normalized = str(plate).strip().upper()
    return normalized or None


def normalize_event_type(camera):
    role = camera.get("camera_role")

    if role in (0, "0", "ENTRY", "entry"):
        return "IN"

    if role in (1, "1", "EXIT", "exit"):
        return "OUT"

    return "DETECTED"


def normalize_bbox(bbox):
    if not bbox or len(bbox) < 4:
        return None

    x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
    return {
        "x": x1,
        "y": y1,
        "w": max(0, x2 - x1),
        "h": max(0, y2 - y1),
    }


def normalize_vehicle_type_id(vehicle_type):
    mapping = {
        "motorbike": 1,
        "oto": 2,
        "xe-tai": 3,
        "xe-container": 4,
    }

    key = str(vehicle_type or "unknown").strip().lower()
    return mapping.get(key)


def build_vehicle_event_payload(camera, result):
    plate = normalize_plate(result.get("plate_text"))
    plate_confidence = float(result.get("avg_confidence") or 0.0)
    vehicle_confidence = result.get("vehicle_confidence")

    status = (
        1
        if plate and plate_confidence >= AUTO_APPROVE_PLATE_CONFIDENCE
        else 0
    )

    vehicle_bbox = normalize_bbox(result.get("vehicle_bbox"))
    plate_bbox = normalize_bbox(result.get("bbox"))

    bbox = {}
    if vehicle_bbox:
        bbox["vehicle"] = vehicle_bbox
    if plate_bbox:
        bbox["plate"] = plate_bbox

    return {
        "camera_id": int(camera["id"]),
        "plate": plate,
        "event_type": normalize_event_type(camera),
        "vehicle_type_id": normalize_vehicle_type_id(result.get("vehicle_type")),
        "vehicle_confidence": vehicle_confidence,
        "plate_confidence": plate_confidence,
        "image_path": None,
        "plate_image_path": None,
        "bbox": bbox or None,
        "status": status,
    }


def post_vehicle_event(camera_api_url, payload):
    url = f"{camera_api_url.rstrip('/')}/vehicle-events"
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"[BE EVENT ERROR] HTTP {exc.code}: {detail}")
    except URLError as exc:
        print(f"[BE EVENT ERROR] Cannot connect: {exc.reason}")
    except Exception as exc:
        print(f"[BE EVENT ERROR] {exc}")

    return None
