
import os
import time
import threading
import base64
import hmac
from pathlib import Path
from urllib.request import HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, build_opener
import json
import hashlib
import re
import io
import uuid
import shutil
import signal
from collections import deque
from types import SimpleNamespace
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room
from urllib.parse import quote, urlencode, urlsplit

import firebase_admin
from firebase_admin import credentials, messaging

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from dotenv import load_dotenv

# Load deployment switches before importing camera and KPI configuration
# modules, which read their values at module initialization time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from database import validate_postgres_connection
from insightface.app import FaceAnalysis
from detection_database import DetectionDatabase
from detection_events import DetectionEventManager
from camera.manager import CameraManager, LatestFrameStore
from camera.rtsp_url import build_rtsp_url, redact_rtsp_url
from notifications.email import EmailAlertQueue, EmailConfig
from config.cameras import CAMERAS, enabled_cameras
from roi.camera_rois import get_camera_roi
from fall_detection import FallDetector
from fall_detection.fall_config import (
    FALL_DETECTION_ENABLED,
    FALL_ENABLED,
    FALL_MODEL_PATH,
)
from ppe_detection import PPEDetector
from ppe_detection.ppe_config import (
    HELMET_DETECTION_ENABLED,
    PPE_ANNOTATION_ENABLED,
    PPE_ENABLED,
    PPE_MODEL_PATH,
)
from smoke_fire_detection import SmokeFireDetector
from smoke_fire_detection.smoke_fire_config import (
    FIRE_DETECTION_ENABLED,
    SMOKE_FIRE_ENABLED,
    SMOKE_FIRE_MODEL_PATH,
    SMOKE_FIRE_TEST_MODE,
)
from object_theft import ObjectTheftDetector
from object_theft.config import (
    OBJECT_THEFT_CAMERAS,
    OBJECT_THEFT_CONFIDENCE_THRESHOLD,
    OBJECT_THEFT_ENABLED,
    OBJECT_THEFT_IOU_THRESHOLD,
    OBJECT_THEFT_MODEL_PATH,
    OBJECT_THEFT_ROI_ENABLED,
)
from face_association import (
    face_associated_with_person as _face_associated_with_person,
    face_box_inside_person,
    is_validated_human_face,
    is_validated_unknown_face,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_NVR_IP = os.getenv("NVR_IP", "192.168.191.40")
NVR_USERNAME = "admin"
NVR_PASSWORD = os.getenv("ALCON_PASSWORD", "")

RTSP_PORT = 554
NVR_HTTP_PORT = int(os.getenv("NVR_HTTP_PORT", "80"))
RTSP_PATH = "/Streaming/Chennels"
RTSP_DEBUG = os.getenv("RTSP_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}

RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.50"))
RECOGNITION_CANDIDATE_THRESHOLD = float(
    os.getenv("RECOGNITION_CANDIDATE_THRESHOLD", "0.43")
)
RECOGNITION_MARGIN = float(os.getenv("RECOGNITION_MARGIN", "0.05"))
PROCESS_EVERY_N_FRAMES = 2
DET_SIZE_VALUE = int(os.getenv("DET_SIZE", "800"))
DET_SIZE = (DET_SIZE_VALUE, DET_SIZE_VALUE)
DET_THRESH = float(os.getenv("DET_THRESH", "0.40"))
REGISTRATION_FALLBACK_DET_SIZE_VALUE = int(
    os.getenv("REGISTRATION_FALLBACK_DET_SIZE", "1200")
)
REGISTRATION_FALLBACK_DET_SIZE = (
    REGISTRATION_FALLBACK_DET_SIZE_VALUE,
    REGISTRATION_FALLBACK_DET_SIZE_VALUE,
)
REGISTRATION_FALLBACK_PADDING_RATIO = float(
    os.getenv("REGISTRATION_FALLBACK_PADDING_RATIO", "0.25")
)
UNKNOWN_FACE_MIN_SCORE = float(
    os.getenv("UNKNOWN_FACE_MIN_SCORE", "0.60")
)
FACE_RECOGNITION_MIN_DET_SCORE = float(
    os.getenv("FACE_RECOGNITION_MIN_DET_SCORE", "0.40")
)
FACE_MIN_WIDTH_RATIO = float(os.getenv("FACE_MIN_WIDTH_RATIO", "0.008"))
FACE_MIN_HEIGHT_RATIO = float(os.getenv("FACE_MIN_HEIGHT_RATIO", "0.012"))
FACE_PERSON_ASSOCIATION_ENABLED = (
    os.getenv("FACE_PERSON_ASSOCIATION_ENABLED", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
FACE_PERSON_MIN_CENTER_INSIDE = (
    os.getenv("FACE_PERSON_MIN_CENTER_INSIDE", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
FACE_PERSON_MIN_OVERLAP = float(os.getenv("FACE_PERSON_MIN_OVERLAP", "0.20"))
PERSON_FACE_CROP_PADDING = float(os.getenv("PERSON_FACE_CROP_PADDING", "0.05"))
FACE_ASSOCIATION_DEBUG = (
    os.getenv("FACE_ASSOCIATION_DEBUG", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
REGISTRATION_MIN_FACE_WIDTH_RATIO = float(
    os.getenv("REGISTRATION_MIN_FACE_WIDTH_RATIO", "0.12")
)
REGISTRATION_MIN_FACE_HEIGHT_RATIO = float(
    os.getenv("REGISTRATION_MIN_FACE_HEIGHT_RATIO", "0.12")
)
REGISTRATION_MIN_FACE_PIXELS = int(
    os.getenv("REGISTRATION_MIN_FACE_PIXELS", "80")
)
REGISTRATION_FACE_EDGE_MARGIN = float(
    os.getenv("REGISTRATION_FACE_EDGE_MARGIN", "0.03")
)
REGISTRATION_MIN_DET_SCORE = float(
    os.getenv("REGISTRATION_MIN_DET_SCORE", "0.40")
)
REGISTRATION_MIN_BLUR_SCORE = float(
    os.getenv("REGISTRATION_MIN_BLUR_SCORE", "40.0")
)
REGISTRATION_IDENTICAL_SIMILARITY = float(
    os.getenv("REGISTRATION_IDENTICAL_SIMILARITY", "0.995")
)
REGISTRATION_FACE_CROP_MARGIN = float(
    os.getenv("REGISTRATION_FACE_CROP_MARGIN", "0.25")
)
REGISTRATION_FACE_CROP_MIN_SIZE = int(
    os.getenv("REGISTRATION_FACE_CROP_MIN_SIZE", "112")
)
REGISTRATION_FACE_CROP_MAX_SIZE = int(
    os.getenv("REGISTRATION_FACE_CROP_MAX_SIZE", "1024")
)
REGISTRATION_SAVE_FACE_CROPS = os.getenv(
    "REGISTRATION_SAVE_FACE_CROPS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
KNOWN_IDENTITY_MEMORY_TIMEOUT = float(
    os.getenv("KNOWN_IDENTITY_MEMORY_TIMEOUT", "2.0")
)
PERSON_TRACK_TIMEOUT = float(
    os.getenv("PERSON_TRACK_TIMEOUT", str(KNOWN_IDENTITY_MEMORY_TIMEOUT))
)
PERSON_MODEL = os.getenv("PERSON_MODEL", "yolo11n.pt")
PERSON_CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.45"))
PERSON_IOU = float(os.getenv("PERSON_IOU", "0.45"))
PERSON_DEDUP_IOU = float(os.getenv("PERSON_DEDUP_IOU", "0.50"))

INFERENCE_DEVICE = "cpu"
VEHICLE_MODEL_PATH = "yolo26l.pt"
VEHICLE_CONFIDENCE = 0.45
TWO_WHEELER_CLASSES = {
    "motorcycle"
}
FOUR_WHEELER_CLASSES = {
    "car",
    "bus",
    "truck"
}
VEHICLE_PROCESS_EVERY_N_FRAMES = 2
VEHICLE_DEDUP_IOU = float(os.getenv("VEHICLE_DEDUP_IOU", "0.50"))

AUTH_PASSWORD_HASH_ITERATIONS = int(
    os.getenv("AUTH_PASSWORD_HASH_ITERATIONS", "210000")
)
DEFAULT_ADMIN_USERNAME = os.getenv("ADMIN_LOGIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_LOGIN_PASSWORD", "admin123")
DEFAULT_USER_USERNAME = os.getenv("USER_LOGIN_USERNAME", "user")
DEFAULT_USER_PASSWORD = os.getenv("USER_LOGIN_PASSWORD", "user123")

# Resolve this from the source file, not the process working directory. This
# keeps known faces and finalized detection history on the same server DB even
# when the service is started by Android deployment tooling or a scheduler.
PROJECT_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = PROJECT_DIR / "known_faces"

HOST = "0.0.0.0"
PORT = 5000

CAMERA_CONFIG_PATH = PROJECT_DIR / "config" / "cameras.json"


# ============================================================
# PUBLIC IMAGE CONFIGURATION
# ============================================================

# Images will be saved here
# Keep event images beside the application so image_url records remain valid
# for every mobile login, independent of the process working directory.
ALERT_IMAGE_DIR = PROJECT_DIR / "alert_images"
ALERT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# Change this to your actual public HTTPS domain.
#
# Example:
# PUBLIC_BASE_URL = "https://bhcs.biconnect.in"
#
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "http://115.242.181.34:5000/"
).rstrip("/")


# ============================================================
# STREAMING
# ============================================================

STREAM_JPEG_QUALITY = 65
STREAM_MAX_WIDTH = 800
STREAM_TARGET_FPS = 10
STREAM_LIVE_FPS = float(
    os.getenv("DASHBOARD_STREAM_FPS", os.getenv("SOCKET_LIVE_FPS", "8"))
)
STREAM_JPEG_QUALITY = int(
    os.getenv("DASHBOARD_JPEG_QUALITY", os.getenv("SOCKET_JPEG_QUALITY", "70"))
)
MAIN_FRAME_MAX_AGE_MS = int(os.getenv("MAIN_FRAME_MAX_AGE_MS", "2000"))
MAX_AI_FRAME_AGE_MS = int(os.getenv("MAX_AI_FRAME_AGE_MS", "2000"))
VIDEO_LATENCY_DEBUG = os.getenv("VIDEO_LATENCY_DEBUG", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
MAX_CONSECUTIVE_READ_FAILURES = 5

UNKNOWN_GONE_CLEARANCE = 5.0
MIN_UNKNOWN_PRESENCE = 1.5

# Alert deduplication: prevent duplicate alerts within this time window (seconds)
ALERT_DEDUP_TIME = 10.0
# Consecutive missing ROI frames must span this duration before an exit closes
# an event. It is deliberately independent of the former alert throttle.
EXIT_CONFIRM_SECONDS = float(os.getenv("EXIT_CONFIRM_SECONDS", "1.0"))
# Final API/Firebase image is selected from confirmed exit frame - 6.
EXIT_FRAME_OFFSET = int(os.getenv("EXIT_FRAME_OFFSET", "6"))

MAX_ALERT_HISTORY = 100


# ============================================================
# RTSP OVER TCP
# ============================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "rtsp_flags;prefer_tcp|"
    "max_delay;1000000|"
    "reorder_queue_size;512|"
    "buffer_size;4194304|"
    "stimeout;5000000|"
    "rw_timeout;5000000|"
    "fflags;discardcorrupt|"
    "err_detect;ignore_err|"
    "flush_packets;1"
)


# ============================================================
# APP / SOCKET SETUP
# ============================================================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "false").strip().lower()
    in {"1", "true", "yes", "on"},
)

try:
    validate_postgres_connection()
except RuntimeError as exc:
    print(f"[ERROR] PostgreSQL database connection failed: {exc}")
    raise

CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

latest_frame_lock = threading.Lock()

latest_annotated_frame = None
latest_annotated_frames = {}

camera_running = True
camera_status = "Starting..."
shutdown_event = threading.Event()
camera_states = {}
camera_event_managers = {}
camera_state_lock = threading.Lock()
camera_manager = None
camera_captures = {}
camera_capture_lock = threading.Lock()
camera_frame_store = LatestFrameStore()
model_inference_lock = threading.RLock()
face_inference_lock = model_inference_lock
shutdown_started = False

known_embeddings = {}
# Legacy average-embedding compatibility map used by existing APIs and older code.
known_person_metadata = {}
# Full per-person embedding templates used by live recognition.
known_face_templates = {}
face_confirmation_state = {}
face_app = None
person_detector = None
vehicle_detector = None
detection_events = None
fall_detector = None
ppe_detector = None
smoke_fire_detector = None
object_theft_detector = None

unknown_present = False
unknown_first_seen = 0.0
unknown_last_seen = 0.0
last_alert_time = 0.0

alert_history = deque(maxlen=MAX_ALERT_HISTORY)

connected_clients = 0
email_alert_queue = None


def _enabled_cameras():
    return enabled_cameras()


def _default_camera():

    cameras = _enabled_cameras()

    if cameras:
        return cameras[0]

    return CAMERAS[0]


def _camera_by_id_or_channel(camera_id=None, channel=None):

    if camera_id:
        camera_id = str(camera_id).strip()
        for camera in CAMERAS:
            if camera["camera_id"].lower() == camera_id.lower():
                return camera

    if channel is not None:
        try:
            channel_value = int(channel)
        except (TypeError, ValueError):
            channel_value = None
        if channel_value is not None:
            for camera in CAMERAS:
                if int(camera.get("channel", 0)) == channel_value:
                    return camera

    return _default_camera()


def _camera_room(camera_id):

    return f"camera:{camera_id}"


def _camera_log(camera_id, message):

    print(f"[{camera_id}] {message}")


def _should_run_kpi(camera_id, kpi_name):
    """Central KPI scope: fire/smoke/known/unknown run on all enabled cameras,
    while helmet/fall are restricted to CAM008 only.
    """
    camera_ref = str(camera_id or "").upper()
    restricted_kpis = {"helmet", "fall"}
    if kpi_name in restricted_kpis:
        return camera_ref == "CAM008"
    return True


def _create_camera_state(camera_config):

    now = time.time()

    return {
        "camera_id": camera_config["camera_id"],
        "camera_name": camera_config["name"],
        "status": "Starting...",
        "latest_annotated_frame": None,
        "latest_frame_timestamp": 0.0,
        "frame_counter": 0,
        "connected": False,
        "processing": False,
        "fps": 0.0,
        "unknown_present": False,
        "unknown_first_seen": 0.0,
        "unknown_last_seen": 0.0,
        "last_alert_time": 0.0,
        "latest_update_at": now,
        "lock": threading.Lock(),
    }


def _camera_state(camera_id):

    return camera_states.get(camera_id)


def _camera_snapshot(camera_id=None):

    camera_id = camera_id or _default_camera()["camera_id"]
    state = _camera_state(camera_id)
    if state is None:
        return None
    with state["lock"]:
        frame = state["latest_annotated_frame"]
        status = state["status"]
        camera_name = state["camera_name"]
        timestamp = state["latest_frame_timestamp"]
        connected = state["connected"]
    if frame is None:
        with latest_frame_lock:
            frame = latest_annotated_frames.get(camera_id)
    return {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "status": status,
        "connected": connected,
        "timestamp": timestamp,
        "frame": frame,
    }


def initialize_camera_states():

    global camera_states

    camera_states = {}
    latest_annotated_frames.clear()

    for camera_config in _enabled_cameras():
        camera_states[camera_config["camera_id"]] = _create_camera_state(
            camera_config
        )

    if not camera_states:
        default_camera = _default_camera()
        camera_states[default_camera["camera_id"]] = _create_camera_state(
            default_camera
        )


def _set_camera_state(camera_id, **updates):

    state = camera_states.get(camera_id)
    if state is None:
        return
    with state["lock"]:
        state.update(updates)


def _release_camera_capture(camera_id):
    with camera_capture_lock:
        cap = camera_captures.pop(camera_id, None)
    if cap is not None:
        cap.release()
        _camera_log(camera_id, "RTSP released")



# ============================================================
# FIREBASE INITIALIZATION
# ============================================================

def initialize_firebase():

    try:

        # Prevent duplicate initialization
        if firebase_admin._apps:
            print("[INFO] Firebase already initialized.")
            return

        firebase_json = os.getenv(
            "FIREBASE_SERVICE_ACCOUNT",
            "./pythonai.json"
        )

        if not os.path.exists(firebase_json):
            raise FileNotFoundError(
                f"Firebase service account not found: {firebase_json}"
            )

        cred = credentials.Certificate(firebase_json)

        firebase_admin.initialize_app(cred)

        print("[OK] Firebase initialized.")

    except Exception as e:

        print(
            f"[ERROR] Firebase initialization failed: {e}"
        )

        raise


# ============================================================
# RTSP URL
# ============================================================

def configure_h264_stream(camera_config):

    config_url = (
        f"http://{camera_config['nvr_ip']}:{NVR_HTTP_PORT}/cgi-bin/configManager.cgi?"
        + urlencode({
            "action": "setConfig",
            "Encode[0].MainFormat[0].Video.Codec": "H.264",
            "Encode[0].ExtraFormat[0].Video.Codec": "H.264",
        })
    )

    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(
        None,
        config_url,
        NVR_USERNAME,
        NVR_PASSWORD,
    )
    opener = build_opener(
        HTTPDigestAuthHandler(password_manager)
    )

    try:
        with opener.open(config_url, timeout=10) as response:
            result = response.read().decode("utf-8", errors="replace")
        if "OK" not in result.upper():
            raise RuntimeError(result.strip() or "Camera rejected H.264 configuration")
        _camera_log(camera_config["camera_id"], "Camera streams configured to H.264.")
    except Exception as e:
        _camera_log(camera_config["camera_id"], f"Could not configure camera to H.264: {e}")

# ============================================================
# INSIGHTFACE
# ============================================================

def get_onnx_providers():
    print("[INFO] Using CPU execution provider.")
    return ["CPUExecutionProvider"]


def initialize_face_model():

    global face_app
    global person_detector
    global vehicle_detector

    if YOLO is None:
        raise RuntimeError(
            "Person detection requires ultralytics. "
            "Install dependencies from requirements.txt."
        )

    print(
        f"[INFO] Loading person detector: {PERSON_MODEL}..."
    )
    person_detector = YOLO(PERSON_MODEL)
    print("[OK] Person detector loaded.")

    providers = get_onnx_providers()

    print(
        "[INFO] Loading InsightFace buffalo_s..."
    )
    print(f"[FACE MODEL] model=buffalo_s provider={providers}")

    face_app = FaceAnalysis(
        name="buffalo_s",
        providers=providers
    )
    recognition_model = face_app.models.get("recognition")
    recognition_model_file = Path(
        getattr(recognition_model, "model_file", "")
    ).name.lower()
    if recognition_model_file != "w600k_r50.onnx":
        raise RuntimeError(
            "The buffalo_s model pack must provide its ArcFace "
            "w600k_r50.onnx recognition model; got "
            f"{recognition_model_file or 'no recognition model'}."
        )

    face_app.prepare(
        ctx_id=-1,
        det_size=DET_SIZE,
        det_thresh=DET_THRESH
    )
    print(
        "[FACE MODEL] recognition=ArcFace "
        f"model={recognition_model_file} "
        f"camera_scope=all_enabled_cameras"
    )

    print(
        f"[INFO] Loading vehicle detector: {VEHICLE_MODEL_PATH}..."
    )
    vehicle_detector = YOLO(VEHICLE_MODEL_PATH)
    print("[OK] Vehicle detector loaded.")

    print("[OK] Face model loaded.")
    print(
        "[FACE MODEL] pack=buffalo_s "
        f"recognition=ArcFace({recognition_model_file})"
    )


def initialize_fall_detector():
    """Load the independent pose model once; existing models remain untouched."""
    global fall_detector
    if not FALL_ENABLED:
        print("[INFO] Fall detection disabled by FALL_DETECTION_ENABLED.")
        return
    if YOLO is None:
        raise RuntimeError("Fall detection requires ultralytics.")
    print(f"[INFO] Loading fall pose detector: {FALL_MODEL_PATH}...")
    fall_detector = FallDetector(FALL_MODEL_PATH, YOLO)
    print("[OK] Fall pose detector loaded.")
    print("[FALL] ENABLED | scope=CAM008_ONLY")


def initialize_ppe_detector():
    """Load PPE once and fail closed without affecting established KPIs."""
    global ppe_detector
    if not PPE_ENABLED:
        print("[INFO] PPE detection disabled by HELMET_DETECTION_ENABLED.")
        return
    if YOLO is None:
        print("[WARNING] PPE detection disabled: ultralytics is unavailable.")
        return
    try:
        print(f"[INFO] Loading PPE detector: {PPE_MODEL_PATH}...")
        ppe_detector = PPEDetector(PPE_MODEL_PATH, YOLO)
        actual_classes = list(ppe_detector.class_names.values())
        supported = sorted(set(ppe_detector.supported_classes.values()))
        required = {"Helmet", "Safety Vest", "Gloves", "Mask", "Safety Boots"}
        missing = sorted(required - set(supported))
        print(f"[PPE MODEL] classes={actual_classes}")
        print(f"[PPE MODEL] supported={supported} missing={missing}")
        print("[OK] PPE detector loaded (only confirmed Helmet sessions alert).")
        print("[HELMET] ENABLED | scope=CAM008_ONLY")
    except Exception as error:
        ppe_detector = None
        print(f"[WARNING] PPE detection disabled; model load failed: {error}")


def initialize_smoke_fire_detector():
    """Load the independent smoke/fire model once without interrupting other KPIs."""
    global smoke_fire_detector
    if not SMOKE_FIRE_ENABLED:
        print("[INFO] Smoke/fire detection disabled by FIRE_DETECTION_ENABLED.")
        return
    if YOLO is None:
        print("[WARNING] Smoke/fire detection disabled: ultralytics is unavailable.")
        return
    try:
        print(f"[INFO] Loading smoke/fire detector: {SMOKE_FIRE_MODEL_PATH}...")
        smoke_fire_detector = SmokeFireDetector(SMOKE_FIRE_MODEL_PATH, YOLO)
        print(
            "[OK] Smoke/fire detector loaded "
            f"(test_mode={SMOKE_FIRE_TEST_MODE})."
        )
        print("[FIRE] ENABLED | scope=ALL_ENABLED_CAMERAS")
    except Exception as error:
        smoke_fire_detector = None
        print(f"[WARNING] Smoke/fire detection disabled; model load failed: {error}")


def initialize_object_theft_detector():
    """Load the isolated Object Theft KPI only when it is enabled for CAM008."""
    global object_theft_detector
    if not OBJECT_THEFT_ENABLED:
        print("[OBJECT THEFT] enabled=False")
        object_theft_detector = None
        return
    scopes = ",".join(sorted(OBJECT_THEFT_CAMERAS)) if OBJECT_THEFT_CAMERAS else "CAM008"
    print(f"[OBJECT THEFT] enabled=True camera_scope={scopes}")
    try:
        object_theft_detector = ObjectTheftDetector(
            model_path=OBJECT_THEFT_MODEL_PATH or None,
            model_loader=None,
            confidence=OBJECT_THEFT_CONFIDENCE_THRESHOLD,
            iou=OBJECT_THEFT_IOU_THRESHOLD,
        )
        if object_theft_detector.model is None:
            object_theft_detector = None
            print("[OBJECT THEFT] model_loaded=False camera_scope=CAM008 status=DISABLED")
        else:
            print(
                f"[OBJECT THEFT] model_loaded=True "
                f"model_path={OBJECT_THEFT_MODEL_PATH or 'unset'}"
            )
    except Exception as error:
        object_theft_detector = None
        print(f"[OBJECT THEFT] MODEL LOAD FAILED camera_scope=CAM008 status=DISABLED error={error}")


# ============================================================
# KNOWN FACE DATABASE
# ============================================================

detection_database = DetectionDatabase()


def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,

        
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return embedding

    return embedding / norm


def _open_db():

    KNOWN_FACES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    from database import get_db_connection

    return get_db_connection()


def _ensure_table(conn):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS known_embeddings (
            person TEXT PRIMARY KEY,
            embedding BYTEA,
            image_paths TEXT,
            fingerprint TEXT,
            updated_at DOUBLE PRECISION
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_events (
            id SERIAL PRIMARY KEY,
            detected_at DOUBLE PRECISION NOT NULL,
            known_count INTEGER NOT NULL DEFAULT 0,
            unknown_count INTEGER NOT NULL DEFAULT 0,
            two_wheeler_count INTEGER NOT NULL DEFAULT 0,
            four_wheeler_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS registered_persons (
            id SERIAL PRIMARY KEY,
            registration_id TEXT NOT NULL UNIQUE,
            storage_key TEXT NOT NULL UNIQUE,
            gate_no TEXT NOT NULL,
            employee_id TEXT UNIQUE,
            employee_name TEXT NOT NULL,
            designation TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS registered_face_embeddings (
            id SERIAL PRIMARY KEY,
            registration_id TEXT NOT NULL,
            image_number INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            embedding BYTEA NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            UNIQUE (registration_id, image_number)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_accounts (
            id SERIAL PRIMARY KEY,
            role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fcm_device_tokens (
            id SERIAL PRIMARY KEY,
            account_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
            fcm_token TEXT NOT NULL UNIQUE,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            last_login_at DOUBLE PRECISION,
            last_logout_at DOUBLE PRECISION,
            FOREIGN KEY (account_id) REFERENCES auth_accounts(id)
                ON DELETE CASCADE
        )
        """
    )


def _bootstrap_auth_accounts(conn):

    defaults = (
        ("admin", DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD),
        ("user", DEFAULT_USER_USERNAME, DEFAULT_USER_PASSWORD),
    )
    now = time.time()

    for role, username, password in defaults:
        username = (username or "").strip()
        password = (password or "").strip()
        if not username or not password:
            continue

        existing = conn.execute(
            "SELECT id FROM auth_accounts WHERE role = ? AND username = ?",
            (role, username),
        ).fetchone()
        if existing is not None:
            continue

        conn.execute(
            """
            INSERT INTO auth_accounts (
                role, username, password_hash, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                role,
                username,
                _hash_password(password),
                now,
                now,
            ),
        )


def _hash_password(password):

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        AUTH_PASSWORD_HASH_ITERATIONS,
    )
    return (
        f"pbkdf2_sha256${AUTH_PASSWORD_HASH_ITERATIONS}$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def _verify_password(password, stored_hash):

    try:
        algorithm, rounds, salt_b64, digest_b64 = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        rounds = int(rounds)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(digest_b64.encode("ascii"))
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            rounds,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _json_or_form_payload():

    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


def _normalize_role(role):

    role = (role or "").strip().lower()
    return role if role in {"admin", "user"} else ""


def _auth_account_by_credentials(role, username, password):

    conn = _open_db()
    try:
        _ensure_table(conn)
        row = conn.execute(
            """
            SELECT id, role, username, password_hash
            FROM auth_accounts
            WHERE role = ? AND username = ?
            """,
            (role, username),
        ).fetchone()
        if row is None or not _verify_password(password, row[3]):
            return None
        return row
    finally:
        conn.close()


def _auth_account_by_id(role, account_id):

    conn = _open_db()
    try:
        _ensure_table(conn)
        return conn.execute(
            """
            SELECT id, role, username, password_hash
            FROM auth_accounts
            WHERE role = ? AND id = ?
            """,
            (role, account_id),
        ).fetchone()
    finally:
        conn.close()


def _upsert_fcm_token(account_id, role, fcm_token, is_active=True):

    fcm_token = (fcm_token or "").strip()
    if not fcm_token:
        return

    conn = _open_db()
    try:
        _ensure_table(conn)
        now = time.time()
        existing = conn.execute(
            """
            SELECT id
            FROM fcm_device_tokens
            WHERE fcm_token = ?
            """,
            (fcm_token,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO fcm_device_tokens (
                    account_id, role, fcm_token, is_active,
                    created_at, updated_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    role,
                    fcm_token,
                    1 if is_active else 0,
                    now,
                    now,
                    now if is_active else None,
                ),
            )
        else:
            conn.execute(
                """
                UPDATE fcm_device_tokens
                SET account_id = ?,
                    role = ?,
                    is_active = ?,
                    updated_at = ?,
                    last_login_at = CASE
                        WHEN ? = 1 THEN ?
                        ELSE last_login_at
                    END
                WHERE fcm_token = ?
                """,
                (
                    account_id,
                    role,
                    1 if is_active else 0,
                    now,
                    1 if is_active else 0,
                    now,
                    fcm_token,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _set_fcm_token_active_status(account_id, role, fcm_token, is_active):

    fcm_token = (fcm_token or "").strip()
    if not fcm_token:
        return False

    conn = _open_db()
    try:
        _ensure_table(conn)
        now = time.time()
        cursor = conn.execute(
            """
            UPDATE fcm_device_tokens
            SET is_active = ?,
                updated_at = ?,
                last_login_at = CASE
                    WHEN ? = 1 THEN COALESCE(last_login_at, ?)
                    ELSE last_login_at
                END,
                last_logout_at = CASE
                    WHEN ? = 0 THEN ?
                    ELSE last_logout_at
                END
            WHERE account_id = ? AND role = ? AND fcm_token = ?
            """,
            (
                1 if is_active else 0,
                now,
                1 if is_active else 0,
                now,
                1 if is_active else 0,
                now,
                account_id,
                role,
                fcm_token,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _mark_fcm_token_inactive(fcm_token):

    fcm_token = (fcm_token or "").strip()
    if not fcm_token:
        return

    conn = _open_db()
    try:
        _ensure_table(conn)
        conn.execute(
            """
            UPDATE fcm_device_tokens
            SET is_active = 0,
                updated_at = ?,
                last_logout_at = ?
            WHERE fcm_token = ?
            """,
            (
                time.time(),
                time.time(),
                fcm_token,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _active_fcm_token_rows(role=None):

    conn = _open_db()
    try:
        _ensure_table(conn)
        if role:
            rows = conn.execute(
                """
                SELECT t.fcm_token, t.account_id, t.role
                FROM fcm_device_tokens AS t
                INNER JOIN auth_accounts AS a ON a.id = t.account_id
                WHERE t.is_active = 1 AND t.role = ? AND a.role = ?
                ORDER BY t.updated_at DESC, t.id DESC
                """,
                (role, role),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT t.fcm_token, t.account_id, t.role
                FROM fcm_device_tokens AS t
                INNER JOIN auth_accounts AS a ON a.id = t.account_id
                WHERE t.is_active = 1
                ORDER BY t.updated_at DESC, t.id DESC
                """,
            ).fetchall()
        return rows
    finally:
        conn.close()


def _all_fcm_token_rows(role=None):

    conn = _open_db()
    try:
        _ensure_table(conn)
        if role:
            rows = conn.execute(
                """
                SELECT t.fcm_token, t.account_id, t.role
                FROM fcm_device_tokens AS t
                INNER JOIN auth_accounts AS a ON a.id = t.account_id
                WHERE t.role = ? AND a.role = ?
                ORDER BY t.updated_at DESC, t.id DESC
                """,
                (role, role),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT t.fcm_token, t.account_id, t.role
                FROM fcm_device_tokens AS t
                INNER JOIN auth_accounts AS a ON a.id = t.account_id
                ORDER BY t.updated_at DESC, t.id DESC
                """,
            ).fetchall()
        return rows
    finally:
        conn.close()


def _chunked(items, size=500):

    for index in range(0, len(items), size):
        yield items[index:index + size]


def _is_invalid_fcm_error(error):

    text = str(error).lower()
    invalid_markers = (
        "registration-token-not-registered",
        "registration token is not a valid fcm registration token",
        "requested entity was not found",
        "invalid registration token",
        "not-registered",
        "unregistered",
        "messaging/invalid-registration-token",
    )
    return any(marker in text for marker in invalid_markers)


def _send_fcm_to_active_tokens(
    title,
    body,
    data,
    image_url=None,
    role=None,
):

    if not firebase_admin._apps:
        initialize_firebase()
    rows = _active_fcm_token_rows(role=role)
    tokens = [row[0] for row in rows if row and row[0]]
    if not tokens:
        return {
            "success": True,
            "sent": 0,
            "invalid_tokens": [],
        }

    invalid_tokens = []
    sent = 0
    android_config = messaging.AndroidConfig(priority="high")
    notification = messaging.Notification(
        title=title,
        body=body,
        image=image_url if image_url else None,
    )
    for token_batch in _chunked(tokens, 500):
        message = messaging.MulticastMessage(
            notification=notification,
            android=android_config,
            data={key: str(value) for key, value in data.items()},
            tokens=token_batch,
        )
        try:
            if hasattr(messaging, "send_each_for_multicast"):
                response = messaging.send_each_for_multicast(message)
                sent += int(getattr(response, "success_count", 0))
                for token, response_item in zip(token_batch, response.responses):
                    if not getattr(response_item, "success", False):
                        error = getattr(response_item, "exception", None)
                        if error is not None and _is_invalid_fcm_error(error):
                            invalid_tokens.append(token)
            else:
                for token in token_batch:
                    try:
                        messaging.send(
                            messaging.Message(
                                notification=notification,
                                android=android_config,
                                data={key: str(value) for key, value in data.items()},
                                token=token,
                            )
                        )
                        sent += 1
                    except Exception as error:
                        if _is_invalid_fcm_error(error):
                            invalid_tokens.append(token)
                        print(
                            f"[WARNING] FCM send failed for token: {error}"
                        )
        except Exception as error:
            print(
                f"[ERROR] FCM multicast send failed: {error}"
            )
            continue

    for token in invalid_tokens:
        _mark_fcm_token_inactive(token)

    return {
        "success": True,
        "sent": sent,
        "invalid_tokens": invalid_tokens,
    }


def _send_fcm_to_all_tokens(
    title,
    body,
    data,
    image_url=None,
    role=None,
):

    if not firebase_admin._apps:
        initialize_firebase()
    rows = _all_fcm_token_rows(role=role)
    tokens = [row[0] for row in rows if row and row[0]]
    if not tokens:
        return {
            "success": True,
            "sent": 0,
            "invalid_tokens": [],
        }

    invalid_tokens = []
    sent = 0
    android_config = messaging.AndroidConfig(priority="high")
    notification = messaging.Notification(
        title=title,
        body=body,
        image=image_url if image_url else None,
    )
    for token_batch in _chunked(tokens, 500):
        message = messaging.MulticastMessage(
            notification=notification,
            android=android_config,
            data={key: str(value) for key, value in data.items()},
            tokens=token_batch,
        )
        try:
            if hasattr(messaging, "send_each_for_multicast"):
                response = messaging.send_each_for_multicast(message)
                sent += int(getattr(response, "success_count", 0))
                for token, response_item in zip(token_batch, response.responses):
                    if not getattr(response_item, "success", False):
                        error = getattr(response_item, "exception", None)
                        if error is not None and _is_invalid_fcm_error(error):
                            invalid_tokens.append(token)
            else:
                for token in token_batch:
                    try:
                        messaging.send(
                            messaging.Message(
                                notification=notification,
                                android=android_config,
                                data={key: str(value) for key, value in data.items()},
                                token=token,
                            )
                        )
                        sent += 1
                    except Exception as error:
                        if _is_invalid_fcm_error(error):
                            invalid_tokens.append(token)
                        print(
                            f"[WARNING] FCM send failed for token: {error}"
                        )
        except Exception as error:
            print(
                f"[ERROR] FCM multicast send failed: {error}"
            )
            continue

    for token in invalid_tokens:
        _mark_fcm_token_inactive(token)

    return {
        "success": True,
        "sent": sent,
        "invalid_tokens": invalid_tokens,
    }


def _embedding_blob(embedding):

    buffer = io.BytesIO()
    np.save(
        buffer,
        np.asarray(embedding, dtype=np.float32),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _registered_person_metadata():
    """Return metadata keyed by the existing known_embeddings storage key."""

    conn = _open_db()
    try:
        _ensure_table(conn)
        rows = conn.execute(
            """
            SELECT storage_key, employee_name, employee_id, designation, gate_no
            FROM registered_persons
            """
        ).fetchall()
        return {
            row[0]: {
                "employee_name": row[1],
                "employee_id": row[2],
                "designation": row[3],
                "gate_no": row[4],
            }
            for row in rows
        }
    finally:
        conn.close()


def _registered_employee_image_url(storage_key, filename):
    return (
        f"{PUBLIC_BASE_URL}/known-faces/"
        f"{quote(str(storage_key))}/{quote(str(filename))}"
    )


def _registered_employees_payload():
    conn = _open_db()
    try:
        _ensure_table(conn)
        person_rows = conn.execute(
            """
            SELECT
                registration_id,
                storage_key,
                gate_no,
                employee_id,
                employee_name,
                designation,
                created_at,
                updated_at
            FROM registered_persons
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        embedding_rows = conn.execute(
            """
            SELECT
                registration_id,
                image_number,
                image_path
            FROM registered_face_embeddings
            ORDER BY registration_id ASC, image_number ASC
            """
        ).fetchall()
    finally:
        conn.close()

    images_by_registration = {}
    for row in embedding_rows:
        image_path = Path(row[2])
        storage_key = image_path.parent.name
        images_by_registration.setdefault(row[0], []).append({
            "image_number": int(row[1]),
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_url": _registered_employee_image_url(
                storage_key,
                image_path.name,
            ),
        })

    employees = []
    for row in person_rows:
        registration_id = row[0]
        storage_key = row[1]
        employees.append({
            "registration_id": registration_id,
            "storage_key": storage_key,
            "gate_no": row[2],
            "employee_id": row[3],
            "employee_name": row[4],
            "designation": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "image_urls": [
                item["image_url"]
                for item in images_by_registration.get(registration_id, [])
            ],
            "images": [
                {
                    "image_number": item["image_number"],
                    "image_name": item["image_name"],
                    "image_path": item["image_path"],
                    "image_url": item["image_url"],
                }
                for item in images_by_registration.get(registration_id, [])
            ],
            "images_count": len(
                images_by_registration.get(registration_id, [])
            ),
        })

    return employees


def _registered_employee_payload(registration_id):

    conn = _open_db()
    try:
        _ensure_table(conn)
        person = conn.execute(
            """
            SELECT
                registration_id,
                storage_key,
                gate_no,
                employee_id,
                employee_name,
                designation,
                created_at,
                updated_at
            FROM registered_persons
            WHERE registration_id = ?
            """,
            (registration_id,),
        ).fetchone()
        if person is None:
            return None
        embedding_rows = conn.execute(
            """
            SELECT
                image_number,
                image_path
            FROM registered_face_embeddings
            WHERE registration_id = ?
            ORDER BY image_number ASC
            """,
            (registration_id,),
        ).fetchall()
    finally:
        conn.close()

    storage_key = person[1]
    images = []
    for row in embedding_rows:
        image_path = Path(row[1])
        images.append({
            "image_number": int(row[0]),
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_url": _registered_employee_image_url(
                storage_key,
                image_path.name,
            ),
        })

    return {
        "registration_id": person[0],
        "storage_key": storage_key,
        "gate_no": person[2],
        "employee_id": person[3],
        "employee_name": person[4],
        "designation": person[5],
        "created_at": person[6],
        "updated_at": person[7],
        "image_urls": [item["image_url"] for item in images],
        "images": images,
        "images_count": len(images),
    }


def _save_detection_event(
    known_count,
    unknown_count,
    two_wheeler_count,
    four_wheeler_count,
    detected_at
):

    conn = _open_db()

    try:

        _ensure_table(conn)

        conn.execute(
            """
            INSERT INTO detection_events
            (
                detected_at,
                known_count,
                unknown_count,
                two_wheeler_count,
                four_wheeler_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                detected_at,
                known_count,
                unknown_count,
                two_wheeler_count,
                four_wheeler_count
            )
        )

        conn.commit()

    except Exception as e:

        print(
            f"[ERROR] Detection event database save failed: {e}"
        )

    finally:

        conn.close()


def _compute_fingerprint(image_paths):

    entries = []

    for p in sorted(image_paths):

        try:

            st = p.stat()

            entries.append(
                f"{p.name}:{int(st.st_mtime)}"
            )

        except Exception:

            entries.append(
                f"{p.name}:0"
            )

    return hashlib.sha1(
        "|".join(entries).encode("utf-8")
    ).hexdigest()


def _save_embedding_db(
    person,
    embedding,
    image_paths,
    fingerprint
):

    buf = io.BytesIO()

    np.save(
        buf,
        np.asarray(
            embedding,
            dtype=np.float32
        ),
        allow_pickle=False
    )

    blob = buf.getvalue()

    conn = _open_db()

    try:

        _ensure_table(conn)

        conn.execute(
            """
            INSERT INTO known_embeddings
            (
                person,
                embedding,
                image_paths,
                fingerprint,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (person)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                image_paths = EXCLUDED.image_paths,
                fingerprint = EXCLUDED.fingerprint,
                updated_at = EXCLUDED.updated_at
            """,
            (
                person,
                blob,
                json.dumps(
                    [
                        str(p)
                        for p in image_paths
                    ]
                ),
                fingerprint,
                time.time()
            )
        )

        conn.commit()

    finally:

        conn.close()


def _load_embedding_db(person):

    conn = _open_db()

    try:

        _ensure_table(conn)

        cur = conn.execute(
            """
            SELECT
                embedding,
                image_paths,
                fingerprint
            FROM known_embeddings
            WHERE person = ?
            """,
            (person,)
        )

        row = cur.fetchone()

        if not row:

            return None

        (
            blob,
            image_paths_json,
            fingerprint
        ) = row

        buf = io.BytesIO(blob)

        buf.seek(0)

        arr = np.load(
            buf,
            allow_pickle=False
        )

        return (
            arr.astype(np.float32),
            json.loads(image_paths_json),
            fingerprint
        )

    finally:

        conn.close()


def _load_registered_embeddings_from_db():

    conn = _open_db()

    try:

        _ensure_table(conn)

        rows = conn.execute(
            """
            SELECT
                p.storage_key,
                p.employee_name,
                p.employee_id,
                p.designation,
                p.gate_no,
                e.image_number,
                e.embedding
            FROM registered_persons AS p
            INNER JOIN registered_face_embeddings AS e
                ON e.registration_id = p.registration_id
            ORDER BY p.created_at ASC, e.image_number ASC
            """
        ).fetchall()

    finally:

        conn.close()

    grouped = {}
    for row in rows:
        storage_key = row[0]
        grouped.setdefault(storage_key, {
            "employee_name": row[1],
            "employee_id": row[2],
            "designation": row[3],
            "gate_no": row[4],
            "embeddings": [],
        })

        blob = row[6]
        if not blob:
            continue

        buf = io.BytesIO(blob)
        buf.seek(0)
        try:
            embedding = np.load(buf, allow_pickle=False).astype(np.float32)
        except Exception as error:
            print(
                f"[KNOWN LOAD] invalid database embedding for "
                f"{storage_key} image_{row[5]}: {error}"
            )
            continue
        grouped[storage_key]["embeddings"].append(embedding)

    return grouped


def _validated_template_embedding(embedding, person_key, image_label=None):
    """Normalize one embedding and discard invalid values without killing the app."""
    try:
        array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception as error:
        print(f"[KNOWN LOAD] invalid embedding for {person_key} ({image_label}): {error}")
        return None

    if array.size == 0:
        print(f"[KNOWN LOAD] empty embedding for {person_key} ({image_label})")
        return None
    if not np.all(np.isfinite(array)):
        print(f"[KNOWN LOAD] non-finite embedding for {person_key} ({image_label})")
        return None

    if array.size != 512:
        print(
            f"[KNOWN LOAD] invalid dimension for {person_key} "
            f"({image_label}): {array.size}, expected 512"
        )
        return None

    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        print(f"[KNOWN LOAD] zero-norm embedding for {person_key} ({image_label})")
        return None

    normalized = array / norm
    return normalized.astype(np.float32, copy=False)


def load_known_faces():

    global known_embeddings
    global known_person_metadata
    global known_face_templates

    KNOWN_FACES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    known_embeddings = {}
    known_person_metadata = {}
    known_face_templates = {}

    registered_rows = _load_registered_embeddings_from_db()
    total_templates = 0
    person_count = 0

    for storage_key, data in registered_rows.items():

        embeddings = data.get("embeddings", [])
        if not embeddings:
            print(f"[KNOWN LOAD] skip {storage_key}: no valid embeddings found")
            continue

        valid_templates = []
        for index, embedding in enumerate(embeddings, start=1):
            normalized = _validated_template_embedding(
                embedding,
                storage_key,
                image_label=f"image_{index}",
            )
            if normalized is None:
                continue
            valid_templates.append(normalized)

        if not valid_templates:
            print(f"[KNOWN LOAD] skip {storage_key}: all embeddings invalid")
            continue

        known_face_templates[storage_key] = valid_templates
        known_person_metadata[storage_key] = {
            "employee_name": data.get("employee_name", storage_key),
            "person_name": data.get("employee_name", storage_key),
            "employee_id": data.get("employee_id"),
            "designation": data.get("designation"),
            "gate_no": data.get("gate_no"),
            "storage_key": storage_key,
        }
        # Keep every registered embedding separately for the live matcher.
        known_embeddings[storage_key] = [
            template.astype(np.float32, copy=False) for template in valid_templates
        ]
        total_templates += len(valid_templates)
        person_count += 1
        print(
            f"[KNOWN LOAD] person={storage_key} "
            f"name={known_person_metadata[storage_key]['employee_name']} "
            f"templates={len(valid_templates)} "
            f"dimension={valid_templates[0].shape[0]} status=OK"
        )

    print(f"[FACE CACHE] persons={person_count} templates={total_templates}")
    for person_key, templates in known_face_templates.items():
        dimensions = sorted({int(template.shape[0]) for template in templates})
        employee_name = known_person_metadata[person_key]["employee_name"]
        print(
            f"[FACE CACHE] employee={employee_name} "
            f"templates={len(templates)} dimensions={dimensions} "
            f"valid={len(templates)}"
        )
    print(
        f"[INFO] Loaded {person_count} registered person(s) with "
        f"{total_templates} individual face templates."
    )


def refresh_known_faces_cache(force=False):
    """Reload the live recognition cache from the persisted registration DB."""
    if force or not known_face_templates:
        print("[KNOWN CACHE] Reloading registered faces...")
        try:
            load_known_faces()
        except Exception as error:
            print(f"[WARN] Failed to reload known faces from DB: {error}")
            return False
        print(f"[KNOWN CACHE] Ready: {len(known_face_templates)} persons loaded")
        return bool(known_face_templates)
    return True


def initialize_auth_accounts():

    conn = _open_db()
    try:
        _ensure_table(conn)
        _bootstrap_auth_accounts(conn)
        conn.commit()
    finally:
        conn.close()


def safe_model_inference(operation, *args, **kwargs):
    """Serialize model calls and pin YOLO prediction to CPU."""
    if getattr(operation, "__name__", "") == "predict":
        kwargs["device"] = INFERENCE_DEVICE
    with model_inference_lock:
        return operation(*args, **kwargs)


def safe_face_inference(frame):
    """Serialize InsightFace calls across all camera workers."""
    if face_app is None:
        return []
    with face_inference_lock:
        return face_app.get(frame)


def _map_crop_face_to_full_frame(face, crop_origin, frame_shape):
    """Translate InsightFace crop-local coordinates back to full-frame coordinates."""
    if face is None:
        return face

    crop_x, crop_y = crop_origin
    frame_height, frame_width = frame_shape[:2]
    if not hasattr(face, "bbox"):
        return face

    try:
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(4).copy()
        bbox[0] = np.clip(float(bbox[0]) + crop_x, 0.0, float(frame_width))
        bbox[1] = np.clip(float(bbox[1]) + crop_y, 0.0, float(frame_height))
        bbox[2] = np.clip(float(bbox[2]) + crop_x, 0.0, float(frame_width))
        bbox[3] = np.clip(float(bbox[3]) + crop_y, 0.0, float(frame_height))
        face.bbox = bbox.astype(np.int32)
    except Exception:
        pass

    for attr_name in ("kps", "landmark_2d_106"):
        points = getattr(face, attr_name, None)
        if points is None:
            continue
        try:
            points_array = np.asarray(points, dtype=np.float32)
            if points_array.size == 0:
                continue
            if points_array.ndim == 2:
                shifted = points_array.copy()
                shifted[:, 0] = np.clip(shifted[:, 0] + crop_x, 0.0, float(frame_width))
                shifted[:, 1] = np.clip(shifted[:, 1] + crop_y, 0.0, float(frame_height))
                setattr(face, attr_name, shifted)
        except Exception:
            continue

    return face


def _detect_faces_in_person_crops(frame, person_boxes, person_tracks=None, camera_id=None):
    """Run InsightFace only on YOLO-detected person crops and remap coordinates to full frame."""
    if frame is None or frame.size == 0:
        return []
    if not person_boxes:
        if camera_id is not None:
            _camera_log(
                camera_id,
                "[FACE GATE] No human person detected -> InsightFace SKIPPED",
            )
            _camera_log(camera_id, "[FACE REJECT] reason=NO_HUMAN_PERSON")
        return []

    frame_height, frame_width = frame.shape[:2]
    detected_faces = []
    person_count = 0
    for person_box in person_boxes:
        if person_box is None:
            continue
        try:
            box = np.asarray(person_box, dtype=np.float32).reshape(4)
        except Exception:
            continue
        if box.size != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            continue

        crop_x1 = max(0, int(np.floor(x1)))
        crop_y1 = max(0, int(np.floor(y1)))
        crop_x2 = min(frame_width, int(np.ceil(x2)))
        crop_y2 = min(frame_height, int(np.ceil(y2)))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        track = (
            next(
                (
                    item for item in person_tracks
                    if item.get("matched_this_frame")
                    and np.array_equal(
                        np.asarray(item.get("box", []), dtype=np.int32),
                        box.astype(np.int32),
                    )
                ),
                None,
            )
            if person_tracks is not None
            else None
        )
        inside_roi = _person_track_in_roi(
            track,
            frame_width,
            frame_height,
            camera_id=camera_id,
        )
        track_id = track["track_id"] if track is not None else "none"
        _camera_log(
            camera_id,
            f"[ROI GATE] camera={camera_id} track_id={track_id} "
            f"person_bbox={tuple(int(v) for v in box)} "
            f"inside_roi={inside_roi}",
        )
        if track is None or not inside_roi:
            if track is not None:
                _camera_log(
                    camera_id,
                    f"[ROI GATE] camera={camera_id} track_id={track_id} "
                    "OUTSIDE ROI -> FACE SKIPPED",
                )
                _camera_log(
                    camera_id,
                    f"[FACE REJECT] camera={camera_id} track_id={track_id} "
                    "reason=OUTSIDE_ROI",
                )
            _camera_log(
                camera_id,
                f"[FACE GATE] camera={camera_id} track_id={track_id} "
                "action=SKIP reason=OUTSIDE_ROI",
            )
            continue

        person_count += 1
        _camera_log(
            camera_id,
            f"[ROI GATE] camera={camera_id} track_id={track_id} INSIDE ROI -> FACE ENABLED",
        )

        person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if person_crop.size == 0:
            _camera_log(
                camera_id,
                f"[FACE REJECT] camera={camera_id} track_id={track_id} "
                "reason=INVALID_PERSON_CROP",
            )
            continue

        _camera_log(
            camera_id,
            f"[FACE GATE] camera={camera_id} track_id={track_id} action=PROCESS",
        )
        _camera_log(
            camera_id,
            f"[FACE SOURCE] camera={camera_id} track_id={track_id} "
            "source=YOLO_PERSON_CROP",
        )

        crop_faces = safe_face_inference(person_crop)
        if not crop_faces:
            continue

        for face in crop_faces:
            mapped_face = _map_crop_face_to_full_frame(face, (crop_x1, crop_y1), frame.shape)
            if mapped_face is None or not face_box_inside_person(
                mapped_face.bbox, person_box,
            ):
                _camera_log(
                    camera_id,
                    f"[FACE REJECT] camera={camera_id} track_id={track_id} "
                    "reason=FACE_OUTSIDE_PERSON",
                )
                continue
            if not is_plausible_face_detection(mapped_face):
                _camera_log(
                    camera_id,
                    f"[FACE REJECT] camera={camera_id} track_id={track_id} "
                    "reason=NON_HUMAN_FACE",
                )
                continue
            mapped_face.person_box = np.asarray(person_box, dtype=np.int32)
            mapped_face.crop_origin = (crop_x1, crop_y1)
            mapped_face.person_track_id = track["track_id"]
            mapped_face.face_validation = {
                "valid_person": True,
                "person_class": 0,
                "person_inside_roi": True,
                "person_crop_valid": True,
                "valid_face": True,
                "face_inside_person": True,
                "human_face_valid": True,
                "face_belongs_to_person": True,
                "unknown_face_score_valid": False,
                "source": "YOLO_PERSON_CROP",
            }
            detected_faces.append(mapped_face)

    if not person_count and camera_id is not None:
        _camera_log(
            camera_id,
            "[FACE GATE] No eligible human person inside ROI -> InsightFace SKIPPED",
        )
    if camera_id is not None and detected_faces:
        _camera_log(
            camera_id,
            f"[FACE-CROP] camera={camera_id} face_count={len(detected_faces)} person_crops={person_count}",
        )
    return detected_faces


def safe_registration_face_inference(frame):
    """Detect registration faces with safe retries for close-up portraits."""
    if face_app is None:
        return []

    with face_inference_lock:
        primary_faces = face_app.get(frame)
        primary_diagnostics = _face_runtime_diagnostics()
        print(
            "[REGISTRATION DETECTION] attempt=primary "
            f"detector={primary_diagnostics.get('detector_name')} "
            f"model_input={primary_diagnostics.get('model_input_shape')} "
            f"providers={primary_diagnostics.get('providers', [])} "
            f"det_size={primary_diagnostics.get('det_size', DET_SIZE)} "
            f"det_thresh={primary_diagnostics.get('det_thresh', DET_THRESH):.3f} "
            f"face_count={len(primary_faces)} "
            f"faces={_face_detection_summary(primary_faces)}"
        )
        if primary_faces:
            return primary_faces

        print(
            "[REGISTRATION FALLBACK] primary_faces=0 "
            f"det_size={DET_SIZE} -> fallback_det_size={REGISTRATION_FALLBACK_DET_SIZE}"
        )
        face_app.prepare(
            ctx_id=-1,
            det_size=REGISTRATION_FALLBACK_DET_SIZE,
            det_thresh=DET_THRESH,
        )
        try:
            fallback_faces = face_app.get(frame)
            fallback_diagnostics = _face_runtime_diagnostics()
            print(
                "[REGISTRATION DETECTION] attempt=fallback "
                f"detector={fallback_diagnostics.get('detector_name')} "
                f"model_input={fallback_diagnostics.get('model_input_shape')} "
                f"providers={fallback_diagnostics.get('providers', [])} "
                f"det_size={fallback_diagnostics.get('det_size', REGISTRATION_FALLBACK_DET_SIZE)} "
                f"det_thresh={fallback_diagnostics.get('det_thresh', DET_THRESH):.3f} "
                f"face_count={len(fallback_faces)} "
                f"faces={_face_detection_summary(fallback_faces)}"
            )
            if fallback_faces:
                return fallback_faces

            padded_frame, pad_x, pad_y = _pad_registration_frame(frame)
            print(
                "[REGISTRATION FALLBACK] fallback_faces=0 "
                "preprocessing=reflect_padding "
                f"padding={pad_x}x{pad_y} "
                f"padded_dimensions={padded_frame.shape[1]}x{padded_frame.shape[0]}"
            )
            padded_faces = face_app.get(padded_frame)
            _translate_registration_faces_from_padded_frame(
                padded_faces, pad_x, pad_y,
            )
            print(
                "[REGISTRATION DETECTION] attempt=padded_fallback "
                f"face_count={len(padded_faces)} "
                f"faces={_face_detection_summary(padded_faces)}"
            )
            return padded_faces
        finally:
            # Registration is allowed to use a larger detector input, but the
            # shared live pipeline must return to its production configuration.
            face_app.prepare(
                ctx_id=-1,
                det_size=DET_SIZE,
                det_thresh=DET_THRESH,
            )


def _pad_registration_frame(frame):
    """Add reflected context around a close portrait for detector-only retry."""
    if frame is None or frame.ndim != 3 or not frame.size:
        raise ValueError("Registration image is not a valid colour frame")

    height, width = frame.shape[:2]
    padding_ratio = max(REGISTRATION_FALLBACK_PADDING_RATIO, 0.0)
    pad_x = max(1, int(np.ceil(width * padding_ratio)))
    pad_y = max(1, int(np.ceil(height * padding_ratio)))
    padded = cv2.copyMakeBorder(
        frame,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REFLECT_101,
    )
    return padded, pad_x, pad_y


def _translate_registration_faces_from_padded_frame(faces, pad_x, pad_y):
    """Map real padded-frame detector geometry to the original image space."""
    offset = np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float32)
    landmark_offset = np.array([pad_x, pad_y], dtype=np.float32)
    for face in faces:
        face.bbox = np.asarray(face.bbox, dtype=np.float32) - offset
        landmarks = getattr(face, "kps", None)
        if landmarks is not None:
            face.kps = np.asarray(landmarks, dtype=np.float32) - landmark_offset


def _face_runtime_diagnostics():
    """Return safe InsightFace runtime metadata without exposing credentials."""
    detector = getattr(face_app, "models", {}).get("detection") if face_app else None
    session = getattr(detector, "session", None)
    providers = (
        session.get_providers()
        if session is not None and hasattr(session, "get_providers")
        else []
    )
    return {
        "providers": providers,
        "detector_name": type(detector).__name__ if detector is not None else None,
        "model_input_shape": (
            list(session.get_inputs()[0].shape)
            if session is not None and hasattr(session, "get_inputs")
            and session.get_inputs()
            else None
        ),
        "det_size": getattr(detector, "input_size", DET_SIZE),
        "det_thresh": float(getattr(detector, "det_thresh", DET_THRESH)),
    }


def _face_detection_summary(faces):
    """Return only the detector outputs needed by registration diagnostics."""
    return [
        {
            "bbox": [round(float(value), 2) for value in face.bbox.tolist()],
            "score": round(float(getattr(face, "det_score", 0.0)), 4),
        }
        for face in faces
    ]


def _registration_image_diagnostics(raw, image, orientation):
    """Log source and decoded image properties before Buffalo_s inference."""
    original_size = None
    original_mode = None
    try:
        with Image.open(io.BytesIO(raw)) as uploaded:
            original_size = uploaded.size
            original_mode = uploaded.mode
    except Exception:
        pass
    return {
        "original_dimensions": original_size,
        "original_mode": original_mode,
        "decoded_dimensions": tuple(image.shape[:2]) if image is not None else None,
        "channels": int(image.shape[2]) if image is not None and image.ndim == 3 else 0,
        "dtype": str(image.dtype) if image is not None else None,
        "decode_succeeded": image is not None,
        "exif_orientation": orientation,
    }


def safe_person_inference(frame):
    """Serialize shared YOLO person inference on CPU."""
    if person_detector is None:
        raise RuntimeError("Person detector is not initialized.")
    return safe_model_inference(person_detector.predict, frame, classes=[0], conf=PERSON_CONFIDENCE, iou=PERSON_IOU, verbose=False)


def safe_vehicle_inference(frame):
    """Serialize shared YOLO vehicle inference on CPU."""
    if vehicle_detector is None:
        raise RuntimeError("Vehicle detector is not initialized.")
    return safe_model_inference(vehicle_detector.predict, frame, conf=VEHICLE_CONFIDENCE, verbose=False)


def recognize_face(face, camera_id=None):
    """Compare a live face against every template, ranked by person."""
    face.recognition_embedding_valid = False
    face.recognition_candidate_name = None
    face.recognition_candidate_score = 0.0
    face.recognition_candidate_key = None
    face.recognition_second_score = 0.0
    face.recognition_margin = 0.0
    face.recognition_decision = "UNKNOWN"
    face.recognition_template = None
    if not known_face_templates:
        refresh_known_faces_cache()

    if not known_face_templates:
        print(f"[FACE] UNKNOWN | best_score=0.00 | best_person=None | threshold={RECOGNITION_THRESHOLD:.2f}")
        face.person_id = None
        return "Unknown", 0.0

    try:
        query = normalize_embedding(np.asarray(face.embedding, dtype=np.float32).reshape(-1))
    except Exception as error:
        print(f"[FACE] UNKNOWN | best_score=0.00 | best_person=None | threshold={RECOGNITION_THRESHOLD:.2f} | reason={error}")
        face.person_id = None
        return "Unknown", 0.0

    if query.size == 0 or not np.all(np.isfinite(query)):
        print(f"[FACE] UNKNOWN | best_score=0.00 | best_person=None | threshold={RECOGNITION_THRESHOLD:.2f} | reason=invalid_live_embedding")
        face.person_id = None
        return "Unknown", 0.0

    face.recognition_embedding_valid = True

    person_scores = {}
    person_templates = {}

    for person_key, templates in known_face_templates.items():
        for template_index, template in enumerate(templates):
            score = float(np.dot(query, template))
            if score > person_scores.get(person_key, -1.0):
                person_scores[person_key] = score
                person_templates[person_key] = template_index + 1

    ranked_people = sorted(
        person_scores.items(), key=lambda item: item[1], reverse=True
    )
    best_key, best_score = ranked_people[0] if ranked_people else (None, 0.0)
    second_score = ranked_people[1][1] if len(ranked_people) > 1 else 0.0
    margin = best_score - second_score
    metadata = known_person_metadata.get(
        best_key,
        {"employee_name": best_key, "person_name": best_key, "employee_id": None},
    ) if best_key is not None else {}
    recognized_name = (
        metadata.get("employee_name") or metadata.get("person_name") or best_key
    ) if best_key is not None else None
    face.recognition_candidate_name = recognized_name
    face.recognition_candidate_score = best_score
    face.recognition_candidate_key = best_key
    face.recognition_second_score = second_score
    face.recognition_margin = margin
    face.recognition_template = person_templates.get(best_key)

    print(
        f"[FACE LIVE] camera={camera_id or 'unknown'} "
        f"det_score={float(getattr(face, 'det_score', 0.0)):.3f} "
        f"face_box={getattr(face, 'bbox', None)} "
        f"embedding_dim={query.size} "
        f"best={recognized_name or 'None'} "
        f"template=image_{person_templates.get(best_key, 'None')} "
        f"score={best_score:.4f} second={second_score:.4f} "
        f"margin={margin:.4f} threshold={RECOGNITION_THRESHOLD:.2f}"
    )

    if best_key is not None and best_score >= RECOGNITION_THRESHOLD and margin >= RECOGNITION_MARGIN:
        face.person_id = metadata.get("employee_id")
        face.recognition_decision = "KNOWN"
        print(
            f"[FACE] MATCH | camera={camera_id or 'unknown'} | "
            f"best_person={recognized_name} | best_score={best_score:.4f} | "
            f"second_score={second_score:.4f} | margin={margin:.4f} | "
            f"template={person_templates[best_key]} | decision=KNOWN"
        )
        return recognized_name, best_score

    if best_key is not None and best_score >= RECOGNITION_CANDIDATE_THRESHOLD:
        face.recognition_decision = "PENDING"
        print(
            f"[FACE] CANDIDATE | camera={camera_id or 'unknown'} | "
            f"person={recognized_name} | score={best_score:.4f} | "
            f"second_score={second_score:.4f} | margin={margin:.4f} | decision=PENDING"
        )
        face.person_id = None
        return "Unknown", best_score

    print(
        f"[FACE] UNKNOWN | camera={camera_id or 'unknown'} | "
        f"best_person={recognized_name or 'None'} | "
        f"best_score={best_score:.4f} | threshold={RECOGNITION_THRESHOLD:.2f}"
    )

    face.person_id = None
    return "Unknown", max(best_score, 0.0)


def boxes_overlap(
    first_box,
    second_box
):

    first_x1, first_y1, first_x2, first_y2 = first_box

    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_width = max(
        0,
        min(first_x2, second_x2)
        -
        max(first_x1, second_x1)
    )

    intersection_height = max(
        0,
        min(first_y2, second_y2)
        -
        max(first_y1, second_y1)
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    first_area = (
        max(
            0,
            first_x2 - first_x1
        )
        *
        max(
            0,
            first_y2 - first_y1
        )
    )

    second_area = (
        max(
            0,
            second_x2 - second_x1
        )
        *
        max(
            0,
            second_y2 - second_y1
        )
    )

    union_area = (
        first_area
        +
        second_area
        -
        intersection_area
    )

    return (
        union_area > 0
        and
        intersection_area / union_area >= 0.10
    )


def _box_center(box):
    return ((float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0)


def _same_face_track(current_box, remembered_box):
    """Require meaningful continuity before retaining a known identity."""
    current = np.asarray(current_box, dtype=np.float32)
    remembered = np.asarray(remembered_box, dtype=np.float32)
    if current.size != 4 or remembered.size != 4:
        return False
    iou = _box_iou(current, remembered)
    current_center = _box_center(current)
    remembered_center = _box_center(remembered)
    current_scale = max(current[2] - current[0], current[3] - current[1], 1.0)
    center_distance = float(np.hypot(
        current_center[0] - remembered_center[0],
        current_center[1] - remembered_center[1],
    ))
    return iou >= 0.20 or center_distance <= current_scale * 0.75


def _match_person_track(person_tracks, box):
    """Find the closest active body track using the existing continuity rule."""
    best_track = None
    best_score = -1.0
    current = np.asarray(box, dtype=np.float32)
    for track in person_tracks:
        previous = np.asarray(track["box"], dtype=np.float32)
        iou = _box_iou(current, previous)
        current_center = _box_center(current)
        previous_center = _box_center(previous)
        scale = max(
            current[2] - current[0],
            current[3] - current[1],
            previous[2] - previous[0],
            previous[3] - previous[1],
            1.0,
        )
        distance = float(np.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        ))
        if iou < 0.10 and distance > max(100.0, scale * 1.5):
            continue
        score = iou - distance / (scale * 10.0)
        if score > best_score:
            best_track = track
            best_score = score
    return best_track


def _update_person_tracks(person_tracks, person_boxes, now, next_track_id):
    """Associate current YOLO bodies to camera-local identity tracks."""
    for track in person_tracks:
        track["matched_this_frame"] = False
    for box in person_boxes:
        track = _match_person_track(
            [item for item in person_tracks if not item["matched_this_frame"]],
            box,
        )
        if track is None:
            body_box = np.asarray(box, dtype=np.int32)
            track = {
                "track_id": next_track_id,
                # ``box`` / ``bbox`` remain the YOLO body box for association
                # and ROI logic. They are strictly internal and never rendered.
                "box": body_box,
                "bbox": body_box.copy(),
                "body_box": body_box.copy(),
                "face_box": None,
                "last_seen": now,
                "matched_this_frame": True,
                "identity_status": "UNVERIFIED",
                "employee_id": None,
                "employee_name": None,
                "last_verified_score": 0.0,
                "annotation_box": None,
                "recognition_score": 0.0,
                "unknown_evidence": False,
            }
            person_tracks.append(track)
            next_track_id += 1
        else:
            body_box = np.asarray(box, dtype=np.int32)
            track.update({
                "box": body_box,
                "bbox": body_box.copy(),
                "body_box": body_box.copy(),
                "last_seen": now,
                "matched_this_frame": True,
            })
    person_tracks[:] = [
        track
        for track in person_tracks
        if now - track["last_seen"] < PERSON_TRACK_TIMEOUT
    ]
    return next_track_id


def _box_iou(first_box, second_box):
    left = max(float(first_box[0]), float(second_box[0]))
    top = max(float(first_box[1]), float(second_box[1]))
    right = min(float(first_box[2]), float(second_box[2]))
    bottom = min(float(first_box[3]), float(second_box[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first_box[2]) - float(first_box[0])) * max(0.0, float(first_box[3]) - float(first_box[1]))
    second_area = max(0.0, float(second_box[2]) - float(second_box[0])) * max(0.0, float(second_box[3]) - float(second_box[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def deduplicate_person_boxes(boxes, scores=None, iou_threshold=None):
    """Keep one highest-confidence YOLO box for each same-frame person."""
    if not boxes:
        return []
    threshold = PERSON_DEDUP_IOU if iou_threshold is None else float(iou_threshold)
    normalized_scores = list(scores or [0.0] * len(boxes))
    candidates = sorted(
        zip(boxes, normalized_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    kept = []
    for box, score in candidates:
        if any(_box_iou(box, kept_box) >= threshold for kept_box, _ in kept):
            continue
        kept.append((np.asarray(box, dtype=np.int32), score))
    return [box for box, _ in kept]


def deduplicate_vehicle_detections(vehicles, iou_threshold=None):
    """Suppress duplicate same-frame vehicle boxes without merging classes."""
    if not vehicles:
        return []
    threshold = VEHICLE_DEDUP_IOU if iou_threshold is None else float(iou_threshold)
    kept = []
    for vehicle in sorted(
        vehicles,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    ):
        if any(
            vehicle["vehicle_type"] == existing["vehicle_type"]
            and _box_iou(vehicle["box"], existing["box"]) >= threshold
            for existing in kept
        ):
            continue
        kept.append(vehicle)
    return kept


def face_inside_person(face_box, person_box):

    face_x1, face_y1, face_x2, face_y2 = face_box
    person_x1, person_y1, person_x2, person_y2 = person_box

    face_center_x = (face_x1 + face_x2) / 2
    face_center_y = (face_y1 + face_y2) / 2

    return (
        person_x1 <= face_center_x <= person_x2
        and person_y1 <= face_center_y <= person_y2
    )


def face_in_roi(face_box, frame_width, frame_height, camera_id=None):

    x1, y1, x2, y2 = face_box

    face_center_x = (x1 + x2) / 2.0
    face_center_y = (y1 + y2) / 2.0
    face_bottom_y = float(y2)

    poly_px = (
        get_camera_roi(camera_id)["points"]
        * np.array([frame_width, frame_height], dtype=np.float32)
    ).astype(np.int32)

    in_center = (
        cv2.pointPolygonTest(
            poly_px,
            (float(face_center_x), float(face_center_y)),
            False,
        )
        >= 0
    )
    in_bottom = (
        cv2.pointPolygonTest(
            poly_px,
            (float(face_center_x), face_bottom_y),
            False,
        )
        >= 0
    )

    return in_center or in_bottom


def _person_track_in_roi(person_track, frame_width, frame_height, camera_id=None):
    """Use the existing person ROI and the tracked body center as the gate."""
    if person_track is None:
        return False

    box = np.asarray(person_track.get("box", []), dtype=np.float32).reshape(-1)
    if box.size != 4:
        return False

    roi_points = (
        get_camera_roi(camera_id)["points"]
        * np.array([frame_width, frame_height], dtype=np.float32)
    )
    if roi_points.shape[0] < 3:
        return False

    center = (
        (float(box[0]) + float(box[2])) / 2.0,
        (float(box[1]) + float(box[3])) / 2.0,
    )
    return cv2.pointPolygonTest(roi_points, center, False) >= 0


def is_plausible_face_detection(face):
    """Reject object regions whose InsightFace landmarks are not face-like."""
    landmarks = np.asarray(getattr(face, "kps", []), dtype=np.float32)
    box = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
    if landmarks.shape != (5, 2) or box.size != 4:
        return False
    if not np.all(np.isfinite(landmarks)) or not np.all(np.isfinite(box)):
        return False

    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    if width / height < 0.55 or width / height > 1.30:
        return False
    if np.any(landmarks[:, 0] < x1 - width * 0.15) or np.any(landmarks[:, 0] > x2 + width * 0.15):
        return False
    if np.any(landmarks[:, 1] < y1 - height * 0.15) or np.any(landmarks[:, 1] > y2 + height * 0.15):
        return False

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < width * 0.30 or eye_distance > width * 0.85:
        return False
    if abs(float(left_eye[1] - right_eye[1])) > height * 0.30:
        return False
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
    if not eye_y < nose[1] < mouth_y:
        return False
    mouth_width = abs(float(right_mouth[0] - left_mouth[0]))
    if mouth_width < width * 0.18 or mouth_width > width * 0.90:
        return False
    if nose[1] - eye_y < height * 0.12 or mouth_y - nose[1] < height * 0.08:
        return False
    eye_left, eye_right = sorted((left_eye[0], right_eye[0]))
    if nose[0] < eye_left - eye_distance * 0.60 or nose[0] > eye_right + eye_distance * 0.60:
        return False
    return True


def face_associated_with_person(face_box, person_boxes, frame_shape):
    """Dispatch to the strict face-person association gate used before recognition."""
    return _face_associated_with_person(face_box, person_boxes, frame_shape)


def vehicle_in_roi(vehicle_box, frame_width, frame_height, camera_id=None):

    x1, y1, x2, y2 = vehicle_box

    vehicle_center_x = (x1 + x2) / 2.0
    vehicle_center_y = (y1 + y2) / 2.0
    vehicle_bottom_y = float(y2)
    vehicle_roi = get_camera_roi(camera_id)["vehicle"]
    polygon = (
        np.asarray(vehicle_roi["points"], dtype=np.float32)
        * np.array([frame_width, frame_height], dtype=np.float32)
    ).astype(np.float32)
    if polygon.shape[0] < 3:
        return False

    center_inside = cv2.pointPolygonTest(
        polygon,
        (float(vehicle_center_x), float(vehicle_center_y)),
        False,
    ) >= 0
    bottom_center_inside = cv2.pointPolygonTest(
        polygon,
        (float(vehicle_center_x), vehicle_bottom_y),
        False,
    ) >= 0
    return center_inside or bottom_center_inside


def _is_plausible_human_box(box, frame_shape):
    """Reject dogs, clutter, and other non-human detections before alerting."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width <= 0 or box_height <= 0:
        return False

    frame_area = float(width * height)
    box_area = box_width * box_height
    if box_area < 0.0015 * frame_area:
        return False
    if box_area > 0.55 * frame_area:
        return False
    if box_height < 0.08 * height:
        return False
    if box_width < 0.05 * width:
        return False

    aspect = box_width / max(box_height, 1.0)
    if aspect < 0.15 or aspect > 1.8:
        return False

    return True


def detect_person_boxes(frame):

    results = safe_person_inference(frame)

    if not results or results[0].boxes is None:
        return []

    raw_boxes = results[0].boxes.xyxy.cpu().numpy()
    raw_classes = results[0].boxes.cls.cpu().numpy()
    raw_scores = results[0].boxes.conf.cpu().numpy()
    person_boxes = [
        (box.astype(int), float(score))
        for box, class_id, score in zip(raw_boxes, raw_classes, raw_scores)
        if int(class_id) == 0 and _is_plausible_human_box(box, frame.shape)
    ]
    deduplicated = deduplicate_person_boxes(
        [box for box, _ in person_boxes],
        [score for _, score in person_boxes],
    )
    return deduplicated

def _scale_person_boxes(person_boxes, source_shape, target_shape):
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    scale_x = target_width / max(float(source_width), 1.0)
    scale_y = target_height / max(float(source_height), 1.0)
    scaled = []
    for box, score in person_boxes:
        values = np.asarray(box, dtype=np.float32)
        values[[0, 2]] *= scale_x
        values[[1, 3]] *= scale_y
        scaled.append((values.astype(np.int32), score))
    return scaled


def _is_plausible_vehicle_box(box, frame_shape):
    """Reject wall/structural false positives that share a vehicle label but not a vehicle shape."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    area = box_width * box_height
    frame_area = float(width * height)

    if box_width <= 0 or box_height <= 0:
        return False
    if area < 0.0025 * frame_area:
        return False
    if area > 0.60 * frame_area:
        return False

    aspect = box_width / max(box_height, 1.0)
    if aspect > 7.0 or aspect < 0.18:
        return False

    if box_height > 0.85 * height and box_width < 0.45 * width:
        return False
    if box_width > 0.80 * width and box_height < 0.25 * height:
        return False

    return True


def detect_vehicle_boxes(frame):

    results = safe_vehicle_inference(frame)

    if not results or results[0].boxes is None:
        return []

    names = results[0].names
    vehicles = []

    for box, class_id, confidence in zip(
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
    ):
        class_name = str(names[int(class_id)]).strip().lower()
        class_name = class_name.replace("-", " ").replace("_", " ")

        if class_name in {"motorcycle", "motorbike", "two wheeler"}:
            class_name = "motorcycle"
        elif class_name in {"car", "bus", "truck", "four wheeler"}:
            class_name = (
                "car"
                if class_name == "four wheeler"
                else class_name
            )

        if (
            class_name not in TWO_WHEELER_CLASSES
            and class_name not in FOUR_WHEELER_CLASSES
        ):
            continue

        if not _is_plausible_vehicle_box(box, frame.shape):
            continue

        vehicle_type = (
            "two_wheeler"
            if class_name in TWO_WHEELER_CLASSES
            else "four_wheeler"
        )

        vehicles.append({
            "box": box.astype(int),
            "class_name": class_name,
            "vehicle_type": vehicle_type,
            "confidence": float(confidence),
        })

    return deduplicate_vehicle_detections(vehicles)


def annotate_alert_frame(frame, faces):

    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]

    for face in faces:
        annotation_box = getattr(face, "annotation_box", None)
        if annotation_box is None:
            annotation_box = getattr(face, "bbox", None)
        if annotation_box is None:
            continue

        x1, y1, x2, y2 = np.asarray(annotation_box, dtype=np.int32)
        name = getattr(face, "recognized_name", "Unknown")
        score = float(getattr(face, "recognition_score", 0.0))

        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        label = "Unknown" if name == "Unknown" else f"{name}  {score:.2f}"

        x1 = max(0, min(int(x1), frame_width - 1))
        y1 = max(0, min(int(y1), frame_height - 1))
        x2 = max(0, min(int(x2), frame_width - 1))
        y2 = max(0, min(int(y2), frame_height - 1))

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


# ============================================================
# SAVE ALERT IMAGE
# ============================================================

def save_alert_image(frame):

    notification_id = str(
        uuid.uuid4()
    )

    filename = (
        f"{notification_id}.jpg"
    )

    image_path = (
        ALERT_IMAGE_DIR /
        filename
    )

    success = cv2.imwrite(
        str(image_path),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:

        raise RuntimeError(
            "Failed to save alert image"
        )

    image_url = (
        f"{PUBLIC_BASE_URL}"
        f"/alerts/{filename}"
    )

    return (
        notification_id,
        image_url
    )


def _queue_email_alert(event, attachment_path=None):
    if email_alert_queue is None:
        print("[WARNING] Email alert not queued: email delivery is not configured.")
        return False
    return email_alert_queue.submit(event, attachment_path=attachment_path)


# ============================================================
# ALERT HANDLING
# ============================================================

def push_alert(
    message,
    frame,
    gate_name=None,
    camera_id=None,
    camera_name=None
):

    gate_name = gate_name or _default_camera()["name"]

    # --------------------------------------------------------
    # Generate ID and save image
    # --------------------------------------------------------

    try:

        (
            notification_id,
            image_url
        ) = save_alert_image(frame)

    except Exception as e:

        print(
            f"[ERROR] Alert image save failed: {e}"
        )

        notification_id = str(
            uuid.uuid4()
        )

        image_url = ""


    # --------------------------------------------------------
    # Local alert history
    # --------------------------------------------------------

    entry = {
        "id": notification_id,
        "time": time.time(),
        "message": message,
        "gate": gate_name,
        "camera_id": camera_id,
        "camera_name": camera_name or gate_name,
        "image_url": image_url,
        "type": "person_detected"
    }

    alert_history.append(entry)


    # --------------------------------------------------------
    # React Native foreground Socket.IO clients
    # --------------------------------------------------------

    socketio.emit(
        "face_alert",
        entry
    )


    attachment_filename = Path(urlsplit(image_url).path).name if image_url else ""
    return _queue_email_alert(
        {
            "id": notification_id,
            "title": "New Person Detected",
            "message": message,
            "detection_type": "person_detected",
            "detected_at": entry["time"],
            "camera_id": camera_id,
            "camera_name": camera_name or gate_name,
            "gate_name": gate_name,
            "image_url": image_url,
        },
        attachment_path=(
            ALERT_IMAGE_DIR / attachment_filename if attachment_filename else None
        ),
    )


def initialize_email_alerts():
    global email_alert_queue

    config = EmailConfig.from_env()
    if config is None:
        email_alert_queue = None
        print(
            "[WARNING] Email alerts are not configured; "
            "set MAIL_HOST, MAIL_PORT, MAIL_FROM, and ALERT_EMAIL_TO."
        )
        return
    email_alert_queue = EmailAlertQueue(config)
    email_alert_queue.start()


def initialize_detection_events():

    global detection_events
    global camera_event_managers

    camera_event_managers = {}

    for camera_config in _enabled_cameras():
        camera_event_managers[camera_config["camera_id"]] = DetectionEventManager(
            database=detection_database,
            image_dir=ALERT_IMAGE_DIR,
            public_base_url=PUBLIC_BASE_URL,
            socketio=socketio,
            email_sender=_queue_email_alert,
            dedup_seconds=EXIT_CONFIRM_SECONDS,
            exit_frame_offset=EXIT_FRAME_OFFSET,
            camera_id=camera_config["camera_id"],
            camera_name=camera_config["name"],
        )

    default_camera = _default_camera()["camera_id"]
    detection_events = camera_event_managers.get(default_camera)


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker(camera_config, rtsp_url=None):

    camera_id = camera_config["camera_id"]
    camera_name = camera_config["name"]
    gate_name = camera_name
    detection_manager = camera_event_managers.get(camera_id)
    state = camera_states.get(camera_id)

    # CameraManager creates workers exclusively from _enabled_cameras(). The
    # three global switches therefore apply to every enabled worker and never
    # to a disabled camera, without maintaining per-KPI camera lists.
    if fall_detector is not None and FALL_DETECTION_ENABLED:
        scope = "CAM008_ONLY" if _should_run_kpi(camera_id, "fall") else "DISABLED"
        _camera_log(camera_id, f"[FALL] ENABLED | scope={scope}")
    if smoke_fire_detector is not None and FIRE_DETECTION_ENABLED:
        _camera_log(camera_id, "[FIRE] ENABLED | scope=ALL_ENABLED_CAMERAS")
    if ppe_detector is not None and HELMET_DETECTION_ENABLED:
        scope = "CAM008_ONLY" if _should_run_kpi(camera_id, "helmet") else "DISABLED"
        _camera_log(camera_id, f"[HELMET] ENABLED | scope={scope}")

    frame_counter = 0
    camera_status = "Starting..."
    last_alert_time = 0.0
    unknown_present = False
    unknown_first_seen = 0.0
    unknown_last_seen = 0.0

    last_faces = []
    last_event_people = []
    last_vehicles = []
    last_ppe_detections = []
    last_smoke_fire_detections = []

    known_face_memory = []
    person_tracks = []
    next_person_track_id = 1

    unknown_frame_history = deque(maxlen=5)
    max_unknown_count = 0
    unknown_alert_sent = False
    last_detection_log_time = 0.0

    last_emit_time = 0.0
    last_frame_diagnostic_time = 0.0
    last_latency_diagnostic_time = 0.0
    last_rtsp_debug_time = 0.0
    emit_interval = 1.0 / max(STREAM_LIVE_FPS, 1.0)

    while not shutdown_event.is_set():

        cap = None
        try:

            if rtsp_url is None:
                rtsp_url = build_rtsp_url(camera_config)

            _camera_log(
                camera_id,
                (
                    f"Connecting to NVR={camera_config['nvr_ip']} "
                    f"CHANNEL={camera_config['channel']} "
                    f"SUBTYPE={camera_config['subtype']}"
                ),
            )
            if RTSP_DEBUG:
                _camera_log(
                    camera_id,
                    f"[RTSP OPEN] camera={camera_id} channel={camera_config['channel']} "
                    f"subtype={camera_config['subtype']} url={redact_rtsp_url(rtsp_url)}",
                )
            cap = cv2.VideoCapture(
                rtsp_url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
                ]
            )
            with camera_capture_lock:
                camera_captures[camera_id] = cap

            if not cap.isOpened():
                camera_status = "RTSP connection failed"
                _set_camera_state(
                    camera_id,
                    status=camera_status,
                    connected=False,
                    latest_update_at=time.time(),
                )
                _camera_log(camera_id, "Could not open MAIN RTSP stream.")
                _release_camera_capture(camera_id)
                shutdown_event.wait(5)
                continue

            camera_status = "LIVE"
            _set_camera_state(
                camera_id,
                status=camera_status,
                connected=True,
                latest_update_at=time.time(),
            )
            _camera_log(camera_id, "RTSP stream connected.")
            consecutive_read_failures = 0

            while not shutdown_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_read_failures += 1
                    if consecutive_read_failures < MAX_CONSECUTIVE_READ_FAILURES:
                        time.sleep(0.05)
                        continue

                    camera_status = "Stream lost - reconnecting..."
                    _set_camera_state(
                        camera_id,
                        status=camera_status,
                        connected=False,
                        latest_update_at=time.time(),
                    )
                    _camera_log(camera_id, "Frame read failed repeatedly.")
                    break


                consecutive_read_failures = 0
                frame_processing_started_at = time.time()

                main_packet = getattr(cap, "last_packet", None)
                frame_age_ms = (
                    max(0.0, (time.time() - main_packet["capture_timestamp"]) * 1000.0)
                    if main_packet is not None
                    else 0.0
                )
                if (
                    main_packet is not None
                    and frame_age_ms > min(MAIN_FRAME_MAX_AGE_MS, MAX_AI_FRAME_AGE_MS)
                ):
                    _camera_log(
                        camera_id,
                        f"[VIDEO] stale MAIN frame dropped age_ms={frame_age_ms:.0f}",
                    )
                    continue

                if RTSP_DEBUG and (time.time() - last_rtsp_debug_time >= 5.0):
                    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
                    frame_hash = hashlib.md5(small.tobytes()).hexdigest()[:8]
                    _camera_log(
                        camera_id,
                        f"[FRAME CHECK] camera={camera_id} channel={camera_config['channel']} "
                        f"subtype={camera_config['subtype']} hash={frame_hash}",
                    )
                    last_rtsp_debug_time = time.time()

                frame_counter += 1

                if (
                    frame_counter
                    %
                    PROCESS_EVERY_N_FRAMES
                    == 0
                ):

                    try:

                        person_frame = frame
                        person_boxes = detect_person_boxes(person_frame)
                        _camera_log(
                            camera_id,
                            f"[PERSON GATE] camera={camera_id} frame={frame_counter} "
                            f"person_count={len(person_boxes)}",
                        )
                        next_person_track_id = _update_person_tracks(
                            person_tracks,
                            person_boxes,
                            time.time(),
                            next_person_track_id,
                        )

                        # Pose analysis consumes only authoritative person
                        # tracks. The pose model never creates a person or
                        # track, and the existing polygon ROI remains master.
                        if fall_detector is not None and _should_run_kpi(camera_id, "fall"):
                            roi_points = get_camera_roi(camera_id)["points"]
                            roi_polygon = roi_points * np.array(
                                [frame.shape[1], frame.shape[0]], dtype=np.float32
                            )
                            confirmed_falls = fall_detector.process(
                                frame=frame,
                                camera_id=camera_id,
                                person_tracks=person_tracks,
                                roi_polygon=roi_polygon,
                                inference=safe_model_inference,
                                now=time.time(),
                            )
                            for fall in confirmed_falls:
                                if detection_manager is not None:
                                    detection_manager.publish_immediate(
                                        {
                                            "detected_at": time.time(),
                                            "detection_type": "fall_detected",
                                            "track_id": fall.track_id,
                                            "person_name": "Unknown",
                                            "confidence": fall.confidence,
                                            "gate_name": gate_name,
                                            "camera_id": camera_id,
                                            "camera_name": camera_name,
                                            "box": fall.box,
                                            "title": "Fall Detected",
                                            "message": f"Fall detected at {gate_name}",
                                        },
                                        frame,
                                        notify=True,
                                    )

                        # PPE receives authoritative person tracks and the
                        # existing polygon ROI. Only confirmed helmet sessions
                        # use the established event/FCM/evidence path.
                        if ppe_detector is not None and _should_run_kpi(camera_id, "helmet"):
                            roi_points = get_camera_roi(camera_id)["points"]
                            roi_polygon = roi_points * np.array(
                                [frame.shape[1], frame.shape[0]], dtype=np.float32
                            )
                            last_ppe_detections, confirmed_helmets = ppe_detector.process(
                                frame=frame,
                                camera_id=camera_id,
                                person_tracks=person_tracks,
                                roi_polygon=roi_polygon,
                                inference=safe_model_inference,
                                now=time.time(),
                            )
                            for helmet in confirmed_helmets:
                                if detection_manager is not None:
                                    detection_manager.publish_immediate(
                                        {
                                            "detected_at": time.time(),
                                            "detection_type": "helmet_detected",
                                            "track_id": helmet.track_id,
                                            "person_name": "Unknown",
                                            "confidence": helmet.confidence,
                                            "gate_name": gate_name,
                                            "camera_id": camera_id,
                                            "camera_name": camera_name,
                                            "box": helmet.box,
                                            "title": "Helmet Detected",
                                            "message": f"Helmet detected on {camera_id}.",
                                        },
                                        frame,
                                        notify=True,
                                    )
                                    _camera_log(camera_id, f"[PPE] HELMET ALERT_SENT | track={helmet.track_id} | confidence={helmet.confidence:.2f}")

                        if OBJECT_THEFT_ENABLED and camera_id.upper() in OBJECT_THEFT_CAMERAS and object_theft_detector is not None:
                            roi_polygon = None
                            if OBJECT_THEFT_ROI_ENABLED:
                                roi_points = get_camera_roi(camera_id)["points"]
                                roi_polygon = roi_points * np.array(
                                    [frame.shape[1], frame.shape[0]], dtype=np.float32
                                )
                            theft_alerts = object_theft_detector.process(
                                frame=frame,
                                camera_id=camera_id,
                                roi_polygon=roi_polygon,
                                inference=safe_model_inference,
                                now=time.time(),
                            )
                            for theft in theft_alerts:
                                if detection_manager is not None:
                                    detection_manager.publish_immediate(
                                        {
                                            "detected_at": time.time(),
                                            "detection_type": "object_theft",
                                            "track_id": theft.track_id,
                                            "person_name": theft.canonical_class,
                                            "confidence": theft.confidence,
                                            "gate_name": gate_name,
                                            "camera_id": camera_id,
                                            "camera_name": camera_name,
                                            "box": theft.bbox,
                                            "annotation_box": theft.bbox,
                                            "mask": theft.mask,
                                            "object_name": theft.canonical_class,
                                            "session_id": theft.session_id,
                                            "title": "Object Theft Detected",
                                            "message": f"{theft.class_name.title()} theft confirmed at {gate_name}",
                                        },
                                        frame,
                                        notify=True,
                                    )
                            _camera_log(camera_id, f"[OBJECT THEFT] processed | alerts={len(theft_alerts)}")

                        # Smoke/fire is camera-local, uses the existing ROI,
                        # and publishes only temporally-confirmed incidents.
                        if smoke_fire_detector is not None:
                            roi_points = get_camera_roi(camera_id)["points"]
                            roi_polygon = roi_points * np.array(
                                [frame.shape[1], frame.shape[0]], dtype=np.float32
                            )
                            last_smoke_fire_detections, confirmed_incidents = smoke_fire_detector.process(
                                frame=frame,
                                camera_id=camera_id,
                                roi_polygon=roi_polygon,
                                inference=safe_model_inference,
                                now=time.time(),
                            )
                            for incident in confirmed_incidents:
                                incident_label = incident.type.title()
                                _camera_log(
                                    camera_id,
                                    "[SMOKE/FIRE] INCIDENT_CONFIRMED "
                                    f"| id={incident.incident_id} "
                                    f"| type={incident.type} "
                                    f"| confidence={incident.confidence:.2f}",
                                )
                                if detection_manager is not None and not SMOKE_FIRE_TEST_MODE:
                                    detection_manager.publish_immediate(
                                        {
                                            "detected_at": time.time(),
                                            "detection_type": f"{incident.type}_detected",
                                            "incident_id": incident.incident_id,
                                            "person_name": "Unknown",
                                            "confidence": incident.confidence,
                                            "gate_name": gate_name,
                                            "camera_id": camera_id,
                                            "camera_name": camera_name,
                                            "box": incident.box,
                                            "smoke_present": incident.smoke_present,
                                            "fire_present": incident.fire_present,
                                            "title": f"{incident_label} Detected",
                                            "message": f"{incident_label} detected at {gate_name}",
                                        },
                                        frame,
                                        notify=True,
                                    )
                                    _camera_log(
                                        camera_id,
                                        f"[SMOKE/FIRE] {incident_label.upper()} ALERT_SENT "
                                        f"| incident={incident.incident_id}",
                                    )
                                elif SMOKE_FIRE_TEST_MODE:
                                    _camera_log(
                                        camera_id,
                                        "[SMOKE/FIRE] test mode: alert not published "
                                        f"| incident={incident.incident_id}",
                                    )

                        if (
                            camera_config.get("kpi") == "vehicle"
                            and
                            frame_counter
                            %
                            VEHICLE_PROCESS_EVERY_N_FRAMES
                            == 0
                        ):
                            detected_vehicles = detect_vehicle_boxes(frame)
                            last_vehicles = [
                                vehicle
                                for vehicle in detected_vehicles
                                if vehicle_in_roi(
                                    vehicle["box"],
                                    frame.shape[1],
                                    frame.shape[0],
                                    camera_id=camera_id,
                                )
                            ]
                        elif camera_config.get("kpi") != "vehicle":
                            last_vehicles = []

                        vehicle_person_boxes = [
                            person_box
                            for person_box in person_boxes
                            if any(
                                boxes_overlap(person_box, vehicle["box"])
                                for vehicle in last_vehicles
                            )
                        ]

                        detected_faces = _detect_faces_in_person_crops(
                            frame,
                            person_boxes,
                            person_tracks=person_tracks,
                            camera_id=camera_id,
                        )

                        if detected_faces:
                            _camera_log(
                                camera_id,
                                "[FACE] detections="
                                + ", ".join(
                                    f"{float(getattr(face, 'det_score', 0.0)):.3f}"
                                    for face in detected_faces
                                ),
                            )

                        now = time.time()
                        accepted_faces = []
                        for face in detected_faces:
                            det_score = float(getattr(face, "det_score", 0.0))
                            face_box = face.bbox.astype(int)
                            associated_person_box = getattr(face, "person_box", None)
                            if (
                                associated_person_box is None
                                or not face_box_inside_person(
                                    face_box,
                                    associated_person_box,
                                )
                            ):
                                _camera_log(
                                    camera_id,
                                    f"[FACE] REJECTED | camera={camera_id} | "
                                    f"reason=face_outside_person | det_score={det_score:.3f} | "
                                    f"face_box={tuple(int(value) for value in face_box)}",
                                )
                                continue
                            _camera_log(
                                camera_id,
                                f"[FACE ASSOCIATION] camera={camera_id} "
                                f"face_box={tuple(int(value) for value in face_box)} "
                                f"person_box={tuple(int(value) for value in associated_person_box)} "
                                "result=PASS",
                            )
                            active_track_match = any(
                                now - remembered_face["last_seen"]
                                < KNOWN_IDENTITY_MEMORY_TIMEOUT
                                and _same_face_track(
                                    face_box,
                                    remembered_face["box"],
                                )
                                for remembered_face in known_face_memory
                            )
                            if det_score < FACE_RECOGNITION_MIN_DET_SCORE:
                                _camera_log(
                                    camera_id,
                                    f"[FACE] REJECTED | camera={camera_id} | "
                                    f"reason=low_detection_confidence | det_score={det_score:.3f}",
                                )
                                continue
                            if not is_plausible_face_detection(face):
                                _camera_log(
                                    camera_id,
                                    f"[FACE] REJECTED | camera={camera_id} | "
                                    f"reason=invalid_face_geometry | det_score={det_score:.3f}",
                                )
                                continue

                            person_track_id = getattr(face, "person_track_id", None)
                            person_track = next(
                                (
                                    track for track in person_tracks
                                    if track.get("matched_this_frame")
                                    and person_track_id is not None
                                    and track["track_id"] == person_track_id
                                ),
                                None,
                            )
                            if person_track is None:
                                _camera_log(
                                    camera_id,
                                    f"[FACE] REJECTED | camera={camera_id} | "
                                    f"reason=no_person_association | det_score={det_score:.3f}",
                                )
                                continue
                            if (
                                not np.array_equal(
                                    np.asarray(person_track.get("box"), dtype=np.int32),
                                    np.asarray(associated_person_box, dtype=np.int32),
                                )
                                or not _person_track_in_roi(
                                    person_track,
                                    frame.shape[1],
                                    frame.shape[0],
                                    camera_id=camera_id,
                                )
                            ):
                                _camera_log(
                                    camera_id,
                                    f"[FACE REJECT] camera={camera_id} "
                                    f"track_id={person_track['track_id']} "
                                    "reason=FACE_OUTSIDE_PERSON_OR_ROI",
                                )
                                continue

                            if FACE_ASSOCIATION_DEBUG:
                                center_x = (face_box[0] + face_box[2]) / 2.0
                                center_y = (face_box[1] + face_box[3]) / 2.0
                                face_area = max(
                                    (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]),
                                    1e-6,
                                )
                                overlap = (
                                    max(0.0, min(face_box[2], associated_person_box[2]) - max(face_box[0], associated_person_box[0]))
                                    * max(0.0, min(face_box[3], associated_person_box[3]) - max(face_box[1], associated_person_box[1]))
                                    / face_area
                                )
                                _camera_log(
                                    camera_id,
                                    f"[FACE-ASSOCIATION] camera={camera_id} person_track={person_track['track_id']} "
                                    f"person_box={tuple(int(value) for value in associated_person_box)} "
                                    f"face_box={tuple(int(value) for value in face_box)} "
                                    f"face_center=({center_x:.1f},{center_y:.1f}) overlap={overlap:.3f} associated=true",
                                )

                            track_key = (camera_id, int(person_track["track_id"]))
                            current_hits = face_confirmation_state.get(track_key, 0)
                            face_confirmation_state[track_key] = current_hits + 1
                            if face_confirmation_state[track_key] < 2:
                                _camera_log(
                                    camera_id,
                                    f"[FACE] REJECTED | camera={camera_id} | reason=temporal_confirmation_pending | "
                                    f"track={person_track['track_id']} | hits={face_confirmation_state[track_key]}",
                                )
                                continue

                            associated_with_vehicle = (
                                associated_person_box is not None
                                and any(
                                    np.array_equal(
                                        associated_person_box,
                                        vehicle_person_box,
                                    )
                                    for vehicle_person_box in vehicle_person_boxes
                                )
                            )
                            face_roi_pass = face_in_roi(
                                face_box,
                                frame.shape[1],
                                frame.shape[0],
                                camera_id=camera_id,
                            )
                            if not face_roi_pass and not associated_with_vehicle:
                                _camera_log(
                                    camera_id,
                                    f"[FACE ROI] camera={camera_id} result=REJECT "
                                    "reason=outside_roi",
                                )
                                continue
                            face_validation = dict(
                                getattr(face, "face_validation", {})
                            )
                            face_validation.update({
                                "valid_person": True,
                                "person_class": 0,
                                "person_inside_roi": True,
                                "person_crop_valid": True,
                                "valid_face": det_score >= FACE_RECOGNITION_MIN_DET_SCORE,
                                "face_inside_person": True,
                                "human_face_valid": is_plausible_face_detection(face),
                                "face_belongs_to_person": True,
                                "unknown_face_score_valid": (
                                    det_score >= UNKNOWN_FACE_MIN_SCORE
                                ),
                                "source": "YOLO_PERSON_CROP",
                            })
                            if not is_validated_human_face(face_validation):
                                _camera_log(
                                    camera_id,
                                    f"[FACE REJECT] camera={camera_id} "
                                    f"track_id={person_track['track_id']} "
                                    "reason=INVALID_FACE",
                                )
                                continue
                            face.face_validation = face_validation
                            person_track["face_validation"] = face_validation.copy()
                            _camera_log(
                                camera_id,
                                f"[FACE ROI] camera={camera_id} result=PASS"
                                + (" reason=vehicle_context" if associated_with_vehicle and not face_roi_pass else ""),
                            )
                            face.associated_person_box = associated_person_box
                            face.person_associated = associated_person_box is not None
                            face.vehicle_context = associated_with_vehicle
                            accepted_faces.append(face)

                        faces_by_person = {}
                        unassociated_faces = []
                        for face in accepted_faces:
                            associated_box = getattr(face, "associated_person_box", None)
                            if associated_box is None:
                                unassociated_faces.append(face)
                                continue
                            person_key = tuple(int(value) for value in associated_box)
                            current_best = faces_by_person.get(person_key)
                            if current_best is None or float(getattr(
                                face, "det_score", 0.0
                            )) > float(getattr(current_best, "det_score", 0.0)):
                                faces_by_person[person_key] = face
                        last_faces = list(faces_by_person.values())
                        for face in unassociated_faces:
                            if not any(
                                _same_face_track(face.bbox, existing.bbox)
                                for existing in last_faces
                            ):
                                last_faces.append(face)
                        if len(accepted_faces) != len(last_faces):
                            _camera_log(
                                camera_id,
                                f"[FACE ASSOCIATION] camera={camera_id} "
                                f"accepted={len(accepted_faces)} "
                                f"unique_person_faces={len(last_faces)}",
                            )

                        excluded_face_count = len(detected_faces) - len(last_faces)
                        if excluded_face_count:
                            _camera_log(
                                camera_id,
                                f"[FACE] {excluded_face_count} detected face(s) excluded "
                                f"by confidence/ROI (min_det_score={FACE_RECOGNITION_MIN_DET_SCORE:.2f})",
                            )

                        if not detected_faces and now - last_detection_log_time >= 5.0:
                            _camera_log(
                                camera_id,
                                "[FACE] InsightFace detected no face",
                            )
                            last_detection_log_time = now

                        seen_unknown_in_frame = False
                        unknown_count_in_frame = 0


                        # ----------------------------------------
                        # Remove old remembered faces
                        # ----------------------------------------

                        active_memory = []
                        for remembered_face in known_face_memory:
                            if now - remembered_face["last_seen"] < KNOWN_IDENTITY_MEMORY_TIMEOUT:
                                active_memory.append(remembered_face)
                            else:
                                _camera_log(
                                    camera_id,
                                    f"[FACE] IDENTITY_CLEARED | camera={camera_id} | "
                                    f"name={remembered_face['name']} | reason=track_timeout",
                                )
                        known_face_memory[:] = active_memory

                        # ----------------------------------------
                        # Process faces
                        # ----------------------------------------

                        for face in last_faces:

                            person_track_id = getattr(face, "person_track_id", None)
                            person_track = next(
                                (
                                    track for track in person_tracks
                                    if track.get("matched_this_frame")
                                    and person_track_id is not None
                                    and track["track_id"] == person_track_id
                                ),
                                None,
                            )
                            if (
                                person_track is None
                                or not _person_track_in_roi(
                                    person_track,
                                    frame.shape[1],
                                    frame.shape[0],
                                    camera_id=camera_id,
                                )
                            ):
                                continue
                            if not is_validated_human_face(
                                getattr(face, "face_validation", None)
                            ):
                                continue

                            (
                                name,
                                score
                            ) = recognize_face(
                                face,
                                camera_id=camera_id,
                            )
                            person_track["recognition_decision"] = getattr(
                                face,
                                "recognition_decision",
                                "UNKNOWN",
                            )

                            current_box = (
                                face.bbox.astype(int)
                            )
                            face.annotation_box = current_box.copy()
                            current_body_box = getattr(
                                face,
                                "associated_person_box",
                                None,
                            )
                            if current_body_box is None:
                                continue
                            face_is_in_roi = face_in_roi(
                                current_box,
                                frame.shape[1],
                                frame.shape[0],
                                camera_id=camera_id,
                            )


                            candidate_key = getattr(
                                face,
                                "recognition_candidate_key",
                                None,
                            )
                            current_strong_match = (
                                name != "Unknown"
                                and candidate_key is not None
                                and score >= RECOGNITION_THRESHOLD
                            )
                            retained_memory = None
                            if not current_strong_match:
                                for remembered_face in known_face_memory:
                                    if not _same_face_track(
                                        current_box,
                                        remembered_face["box"],
                                    ):
                                        continue
                                    remembered_body_box = remembered_face.get("body_box")
                                    if (
                                        current_body_box is not None
                                        and remembered_body_box is not None
                                        and not _same_face_track(
                                            current_body_box,
                                            remembered_body_box,
                                        )
                                    ):
                                        continue
                                    retained_memory = remembered_face
                                    break

                            if current_strong_match:
                                metadata = known_person_metadata.get(candidate_key, {})
                                if person_track is not None:
                                    person_track.update({
                                        "identity_status": "KNOWN",
                                        "employee_id": metadata.get("employee_id"),
                                        "employee_name": name,
                                        "last_verified_score": score,
                                        "body_box": (
                                            current_body_box.copy()
                                            if current_body_box is not None else None
                                        ),
                                        "bbox": (
                                            current_body_box.copy()
                                            if current_body_box is not None else person_track.get("bbox")
                                        ),
                                        "face_box": current_box.copy(),
                                        "annotation_box": current_box.copy(),
                                        "recognition_score": score,
                                        "unknown_evidence": True,
                                    })
                                matching_memory = next(
                                    (
                                        remembered_face
                                        for remembered_face in known_face_memory
                                        if remembered_face.get("key") == candidate_key
                                        and _same_face_track(
                                            current_box,
                                            remembered_face["box"],
                                        )
                                    ),
                                    None,
                                )
                                if matching_memory is None:
                                    matching_memory = {
                                        "key": candidate_key,
                                        "name": name,
                                        "employee_id": metadata.get("employee_id"),
                                        "score": score,
                                        "last_verified_score": score,
                                        "box": current_box,
                                        "body_box": current_body_box,
                                        "last_seen": now,
                                        "camera_id": camera_id,
                                        "identity_source": "face_match",
                                        "confidence_state": "verified",
                                    }
                                    known_face_memory.append(matching_memory)
                                else:
                                    matching_memory.update({
                                        "name": name,
                                        "employee_id": metadata.get("employee_id"),
                                        "score": score,
                                        "last_verified_score": score,
                                        "box": current_box,
                                        "body_box": current_body_box,
                                        "last_seen": now,
                                        "confidence_state": "verified",
                                    })
                                _camera_log(
                                    camera_id,
                                    f"[FACE] KNOWN | name={name} | "
                                    f"current_similarity={score:.4f} | source=current_match",
                                )
                            elif person_track is not None and person_track.get("identity_status") == "KNOWN":
                                name = person_track["employee_name"]
                                face.person_id = person_track.get("employee_id")
                                person_track.update({
                                    "body_box": (
                                        current_body_box.copy()
                                        if current_body_box is not None
                                        else person_track.get("body_box")
                                    ),
                                    "bbox": (
                                        current_body_box.copy()
                                        if current_body_box is not None else person_track.get("bbox")
                                    ),
                                    "face_box": current_box.copy(),
                                    "annotation_box": current_box.copy(),
                                    "recognition_score": score,
                                    "unknown_evidence": True,
                                })
                                face.recognition_decision = "RETAINED"
                                _camera_log(
                                    camera_id,
                                    f"[IDENTITY] camera={camera_id} "
                                    f"track={person_track['track_id']} "
                                    f"current={getattr(face, 'recognition_candidate_name', 'UNKNOWN')} "
                                    f"established={name} final=KNOWN "
                                    "reason=retain_active_known_track",
                                )
                            elif retained_memory is not None:
                                name = retained_memory["name"]
                                face.person_id = retained_memory.get("employee_id")
                                retained_memory.update({
                                    "box": current_box,
                                    "body_box": (
                                        current_body_box
                                        if current_body_box is not None
                                        else retained_memory.get("body_box")
                                    ),
                                    "last_seen": now,
                                    "score": score,
                                    "confidence_state": "retained",
                                })
                                face.recognition_decision = "RETAINED"
                                _camera_log(
                                    camera_id,
                                    f"[FACE] KNOWN_RETAINED | camera={camera_id} | "
                                    f"name={name} | current_similarity={score:.4f} | "
                                    f"last_verified={retained_memory['last_verified_score']:.4f} | "
                                    "source=active_track",
                                )
                            elif getattr(face, "recognition_decision", None) == "PENDING":
                                name = "Pending"
                                face.person_id = None
                            else:
                                name = "Unknown"
                                face.person_id = None

                            unknown_face_valid = is_validated_unknown_face(
                                face.face_validation,
                                getattr(face, "recognition_decision", None),
                            )
                            if person_track is not None and name == "Unknown" and unknown_face_valid:
                                person_track.update({
                                    "identity_status": "UNKNOWN",
                                    "body_box": (
                                        current_body_box.copy()
                                        if current_body_box is not None else None
                                    ),
                                    "bbox": (
                                        current_body_box.copy()
                                        if current_body_box is not None else person_track.get("bbox")
                                    ),
                                    "face_box": current_box.copy(),
                                    "annotation_box": current_box.copy(),
                                    "recognition_score": score,
                                    "unknown_evidence": True,
                                })

                            face.recognized_name = name
                            face.recognition_state = (
                                "KNOWN"
                                if name not in {"Unknown", "Pending"}
                                else "UNKNOWN"
                                if (
                                    name == "Unknown"
                                    and unknown_face_valid
                                )
                                else "PENDING"
                            )

                            face.recognition_score = score
                            face.in_roi = face_is_in_roi
                            face.unknown_evidence = (
                                face.recognition_state == "UNKNOWN"
                                and unknown_face_valid
                            )

                            face_detection_score = float(
                                getattr(face, "det_score", 0.0)
                            )
                            recognition_decision = getattr(
                                face, "recognition_decision", None
                            )
                            if recognition_decision == "PENDING":
                                unknown_gate_status = "HELD_AMBIGUOUS_KNOWN_MATCH"
                            elif name == "Unknown" and not unknown_face_valid:
                                if face_detection_score < UNKNOWN_FACE_MIN_SCORE:
                                    unknown_gate_status = "REJECTED_LOW_FACE_SCORE"
                                else:
                                    unknown_gate_status = "REJECTED_VALIDATION"
                            elif name == "Unknown" and unknown_face_valid:
                                unknown_gate_status = "ACCEPTED"
                            else:
                                unknown_gate_status = "KNOWN"

                            if (
                                person_track.get("unknown_gate_diagnostic")
                                != unknown_gate_status
                            ):
                                person_track["unknown_gate_diagnostic"] = (
                                    unknown_gate_status
                                )
                                if unknown_gate_status != "KNOWN":
                                    _camera_log(
                                        camera_id,
                                        f"[UNKNOWN GATE] track={person_track['track_id']} "
                                        f"result={unknown_gate_status} "
                                        f"decision={recognition_decision} "
                                        f"face_det_score={face_detection_score:.3f}/"
                                        f"{UNKNOWN_FACE_MIN_SCORE:.3f} "
                                        f"similarity={score:.3f} "
                                        f"candidate_threshold="
                                        f"{RECOGNITION_CANDIDATE_THRESHOLD:.3f} "
                                        f"known_threshold={RECOGNITION_THRESHOLD:.3f} "
                                        f"margin_required={RECOGNITION_MARGIN:.3f}",
                                    )


                            if (
                                name == "Unknown"
                                and face.recognition_state == "UNKNOWN"
                                and face_is_in_roi
                                and face.unknown_evidence
                                and face_detection_score
                                >= UNKNOWN_FACE_MIN_SCORE
                            ):
                                seen_unknown_in_frame = True


                        current_face_track_ids = {
                            int(face.person_track_id)
                            for face in last_faces
                            if getattr(face, "person_track_id", None) is not None
                        }
                        roi_person_tracks = [
                            person_track
                            for person_box in person_boxes
                            for person_track in person_tracks
                            if (
                                person_track.get("matched_this_frame")
                                and int(person_track["track_id"]) in current_face_track_ids
                                and np.array_equal(
                                    np.asarray(person_track.get("box"), dtype=np.int32),
                                    np.asarray(person_box, dtype=np.int32),
                                )
                            )
                        ]
                        last_event_people = []
                        event_person_track_ids = set()
                        for person_track in roi_person_tracks:
                            person_box = np.asarray(
                                person_track["box"], dtype=np.int32,
                            )

                            identity_status = person_track.get("identity_status")
                            if identity_status == "KNOWN":
                                recognition_state = "KNOWN"
                                identity_name = person_track.get("employee_name") or "Unknown"
                            elif (
                                identity_status == "UNKNOWN"
                                and person_track.get("unknown_evidence")
                            ):
                                recognition_state = "UNKNOWN"
                                identity_name = "Unknown"
                            else:
                                recognition_state = "PENDING"
                                identity_name = "Pending"

                            annotation_box = person_track.get("annotation_box")
                            last_event_people.append(
                                SimpleNamespace(
                                    bbox=np.asarray(person_box, dtype=np.int32),
                                    associated_person_box=np.asarray(
                                        person_box, dtype=np.int32
                                    ),
                                    annotation_box=(
                                        tuple(int(value) for value in annotation_box)
                                        if annotation_box is not None
                                        else None
                                    ),
                                    recognized_name=identity_name,
                                    recognition_state=recognition_state,
                                    recognition_decision=person_track.get(
                                        "recognition_decision"
                                    ),
                                    face_validation=person_track.get(
                                        "face_validation"
                                    ),
                                    recognition_score=float(
                                        person_track.get("recognition_score", 0.0)
                                    ),
                                    person_id=person_track.get("employee_id"),
                                    track_id=person_track["track_id"],
                                )
                            )
                            event_person_track_ids.add(person_track["track_id"])

                        unique_unknown_people = [
                            person
                            for person in last_event_people
                            if getattr(person, "recognized_name", "Unknown") == "Unknown"
                        ]
                        unknown_count_in_frame = len(unique_unknown_people)
                        seen_unknown_in_frame = bool(unique_unknown_people)
                        _camera_log(
                            camera_id,
                            f"[UNKNOWN COUNT] camera={camera_id} "
                            f"unique_tracks={unknown_count_in_frame}",
                        )
                        unknown_vehicle_person_detected = any(
                            getattr(face, "recognized_name", "Unknown") == "Unknown"
                            and any(
                                face_inside_person(
                                    face.bbox.astype(int),
                                    person_box,
                                )
                                for person_box in vehicle_person_boxes
                            )
                            for face in last_faces
                        )

                        if now - last_detection_log_time >= 5.0:

                            known_count_in_frame = sum(
                                1
                                for face in last_faces
                                if getattr(
                                    face,
                                    "recognized_name",
                                    "Unknown"
                                ) != "Unknown"
                            )
                            two_wheeler_count = sum(
                                1
                                for vehicle in last_vehicles
                                if vehicle["vehicle_type"] == "two_wheeler"
                            )
                            four_wheeler_count = sum(
                                1
                                for vehicle in last_vehicles
                                if vehicle["vehicle_type"] == "four_wheeler"
                            )

                            if (
                                known_count_in_frame
                                or unknown_count_in_frame
                                or two_wheeler_count
                                or four_wheeler_count
                            ):
                                _save_detection_event(
                                    known_count_in_frame,
                                    unknown_count_in_frame,
                                    two_wheeler_count,
                                    four_wheeler_count,
                                    now
                                )

                            vehicle_labels = ", ".join(
                                f"{vehicle['vehicle_type']}"
                                f" ({vehicle['class_name']}, "
                                f"{vehicle['confidence']:.2f})"
                                for vehicle in last_vehicles
                            ) or "none"

                            _camera_log(
                                camera_id,
                                f"Faces detected: {len(detected_faces)}, "
                                f"persons: {len(person_boxes)}, "
                                f"inside ROI: {len(last_faces)}, "
                                f"known: {known_count_in_frame}, "
                                f"unknown: {unknown_count_in_frame}, "
                                f"two_wheeler: {two_wheeler_count}, "
                                f"four_wheeler: {four_wheeler_count}, "
                                f"vehicles: {vehicle_labels}",
                            )

                            for vehicle in last_vehicles:
                                _camera_log(
                                    camera_id,
                                    f"Vehicle detected: {vehicle['vehicle_type']} "
                                    f"({vehicle['class_name']}), confidence: "
                                    f"{vehicle['confidence']:.2f}",
                                )

                            last_detection_log_time = now


                        # ----------------------------------------
                        # Unknown person presence
                        # ----------------------------------------

                        if seen_unknown_in_frame:

                            if not unknown_present:

                                unknown_present = True

                                unknown_first_seen = now

                            unknown_frame_history.append(
                                annotate_alert_frame(
                                    frame,
                                    last_faces,
                                )
                            )
                            max_unknown_count = max(
                                max_unknown_count,
                                unknown_count_in_frame,
                            )

                            if unknown_present:

                                unknown_last_seen = now

                                if (
                                    not unknown_alert_sent
                                    and (
                                        unknown_vehicle_person_detected
                                        or now - unknown_first_seen
                                        >= MIN_UNKNOWN_PRESENCE
                                    )
                                ):

                                    if (
                                        now - last_alert_time
                                        >= ALERT_DEDUP_TIME
                                    ):

                                        _camera_log(camera_id, "Unknown person detected in ROI.")

                                        camera_status = "Unknown person detected"
                                        _set_camera_state(
                                            camera_id,
                                            status=camera_status,
                                            latest_update_at=time.time(),
                                        )

                                        _camera_log(camera_id, "Unknown event will be saved by the detection manager.")

                                        last_alert_time = now

                                    else:

                                        _camera_log(
                                            camera_id,
                                            "Alert suppressed "
                                            f"(duplicate within {ALERT_DEDUP_TIME}s window)",
                                        )

                                    unknown_alert_sent = True


                        # ----------------------------------------
                        # Person left frame
                        # ----------------------------------------

                        elif (
                            unknown_present
                            and
                            now
                            -
                            unknown_last_seen
                            >=
                            UNKNOWN_GONE_CLEARANCE
                        ):

                            unknown_present = False

                            unknown_first_seen = 0.0

                            unknown_last_seen = 0.0

                            unknown_frame_history.clear()

                            max_unknown_count = 0

                            unknown_alert_sent = False


                    except Exception as e:

                        _camera_log(camera_id, f"[ERROR] Face processing error: {e}")


                height, width = frame.shape[:2]
                # ------------------------------------------------
                # ANNOTATED FRAME
                # ------------------------------------------------

                now = time.time()


                if (
                    now
                    -
                    last_emit_time
                    >=
                    emit_interval
                ):

                    last_emit_time = now

                    send_frame = frame.copy()

                    if ppe_detector is not None and PPE_ANNOTATION_ENABLED and _should_run_kpi(camera_id, "helmet"):
                        ppe_detector.annotate(send_frame, last_ppe_detections)

                    if smoke_fire_detector is not None:
                        smoke_fire_detector.annotate(send_frame, last_smoke_fire_detections)

                    for face in last_faces:

                        if (
                            getattr(face, "recognized_name", "Unknown") == "Unknown"
                            and not getattr(face, "unknown_evidence", False)
                        ):
                            continue

                        x1, y1, x2, y2 = (
                            face.bbox.astype(int)
                        )

                        name = getattr(
                            face,
                            "recognized_name",
                            "Unknown"
                        )

                        score = float(
                            getattr(
                                face,
                                "recognition_score",
                                0.0
                            )
                        )


                        x1 = max(
                            0,
                            min(
                                x1,
                                width - 1
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                y1,
                                height - 1
                            )
                        )

                        x2 = max(
                            0,
                            min(
                                x2,
                                width - 1
                            )
                        )

                        y2 = max(
                            0,
                            min(
                                y2,
                                height - 1
                            )
                        )


                        if name == "Unknown":

                            color = (
                                0,
                                0,
                                255
                            )

                            label = "Unknown"

                        else:

                            color = (
                                0,
                                255,
                                0
                            )

                            label = (
                                f"{name} "
                                f"{score:.2f}"
                            )


                        cv2.rectangle(
                            send_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )


                        (
                            text_width,
                            text_height
                        ), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            2
                        )


                        label_y1 = max(
                            0,
                            y1
                            -
                            text_height
                            -
                            baseline
                            -
                            8
                        )


                        label_y2 = max(
                            text_height
                            +
                            baseline
                            +
                            8,
                            y1
                        )


                        cv2.rectangle(
                            send_frame,
                            (x1, label_y1),
                            (
                                x1
                                +
                                text_width
                                +
                                10,
                                label_y2
                            ),
                            color,
                            -1
                        )


                        cv2.putText(
                            send_frame,
                            label,
                            (
                                x1 + 5,
                                label_y2 - 6
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (
                                255,
                                255,
                                255
                            ),
                            2,
                            cv2.LINE_AA
                        )


                    for vehicle in last_vehicles:
                        x1, y1, x2, y2 = vehicle["box"]
                        x1 = max(0, min(int(x1), width - 1))
                        y1 = max(0, min(int(y1), height - 1))
                        x2 = max(0, min(int(x2), width - 1))
                        y2 = max(0, min(int(y2), height - 1))
                        label = (
                            f"Vehicle detected: "
                            f"{vehicle['vehicle_type']} "
                            f"{vehicle['confidence']:.2f}"
                        )
                        color = (
                            (255, 165, 0)
                            if vehicle["vehicle_type"] == "two_wheeler"
                            else (255, 0, 0)
                        )

                        cv2.rectangle(
                            send_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )
                        cv2.putText(
                            send_frame,
                            label,
                            (x1 + 5, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                            cv2.LINE_AA
                        )


                    if detection_manager is not None:
                        # The tracker retains frames until a detection exits the
                        # ROI, then draws that detection once on the selected
                        # frame.  Supplying send_frame here would retain the
                        # preview overlays as well, so the final draw produced
                        # stale/misplaced annotations and a second vehicle box.
                        detection_manager.process_frame(
                            frame=frame,
                            faces=last_event_people,
                            vehicles=last_vehicles,
                            gate_name=gate_name,
                            detected_at=now,
                            alert_frame=frame,
                        )


                    # --------------------------------------------
                    # Resize
                    # --------------------------------------------

                    if width > STREAM_MAX_WIDTH:

                        scale = (
                            STREAM_MAX_WIDTH
                            /
                            float(width)
                        )

                        send_frame = cv2.resize(
                            send_frame,
                            (
                                STREAM_MAX_WIDTH,
                                int(
                                    height * scale
                                )
                            )
                        )


                    # --------------------------------------------
                    # JPEG encode
                    # --------------------------------------------

                    jpeg_started_at = time.time()
                    success, encoded = cv2.imencode(
                        ".jpg",
                        send_frame,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            STREAM_JPEG_QUALITY
                        ]
                    )


                    if success:

                        encoded_bytes = (
                            encoded.tobytes()
                        )

                        if now - last_frame_diagnostic_time >= 5.0:
                            _camera_log(
                                camera_id,
                                f"Frame generated: {width}x{height}, "
                                f"jpeg={len(encoded_bytes) // 1024}KB, "
                                f"base64={((len(encoded_bytes) + 2) // 3) * 4} chars",
                            )
                            last_frame_diagnostic_time = now


                        if state is not None:
                            with state["lock"]:
                                state["latest_annotated_frame"] = encoded_bytes
                                state["latest_frame_timestamp"] = now
                                state["status"] = camera_status
                                state["frame_counter"] = frame_counter
                                state["processing"] = True
                                state["connected"] = True

                        with latest_frame_lock:
                            latest_annotated_frame = encoded_bytes
                            latest_annotated_frames[camera_id] = encoded_bytes


                        if VIDEO_LATENCY_DEBUG and now - last_latency_diagnostic_time >= 5.0:
                            _camera_log(
                                camera_id,
                                "[VIDEO-LATENCY] "
                                f"stream=MAIN frame_id={main_packet['frame_id'] if main_packet else frame_counter} "
                                f"ai_ms={(jpeg_started_at - frame_processing_started_at) * 1000.0:.1f} "
                                f"jpeg_ms={(time.time() - jpeg_started_at) * 1000.0:.1f}",
                            )
                            last_latency_diagnostic_time = now


            if cap is not None:
                _release_camera_capture(camera_id)
            if not shutdown_event.is_set():
                shutdown_event.wait(5)
        except Exception as e:

            camera_status = (
                f"Error: {e}"
            )
            _set_camera_state(
                camera_id,
                status=camera_status,
                connected=False,
                latest_update_at=time.time(),
            )

            _camera_log(camera_id, f"[ERROR] Camera worker: {e}")

            if cap is not None:
                _release_camera_capture(camera_id)
            shutdown_event.wait(5)


# ============================================================
# ANDROID PERSON REGISTRATION
# ============================================================

def _decode_registration_image(raw):
    """Decode an upload and apply EXIF orientation before face inference."""
    if not raw:
        return None, None
    try:
        with Image.open(io.BytesIO(raw)) as uploaded:
            orientation = uploaded.getexif().get(274)
            corrected = ImageOps.exif_transpose(uploaded).convert("RGB")
            rgb = np.asarray(corrected)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), orientation
    except Exception:
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        return image, None


def _validate_registration_images():
    """Validate five uploads and embed only their validated face crops."""

    expected = [f"image{number}" for number in range(1, 6)]
    supplied = [
        field
        for field in request.files
        if field.lower().startswith("image")
    ]
    if set(supplied) != set(expected) or any(
        len(request.files.getlist(field)) != 1
        for field in expected
    ):
        raise ValueError("Exactly 5 face images are required")
    if face_app is None:
        raise RuntimeError("Face recognition model is not ready")

    images, crops, embeddings = [], [], []
    for number, field in enumerate(expected, start=1):
        raw = request.files[field].read()
        image, orientation = _decode_registration_image(raw)
        image_diagnostics = _registration_image_diagnostics(
            raw,
            image,
            orientation,
        )
        runtime_diagnostics = _face_runtime_diagnostics()
        print(
            f"[REGISTRATION IMAGE] image={number} bytes={len(raw)} "
            f"original_dimensions={image_diagnostics['original_dimensions']} "
            f"original_mode={image_diagnostics['original_mode']} "
            f"decoded_dimensions={image_diagnostics['decoded_dimensions']} "
            f"channels={image_diagnostics['channels']} "
            f"dtype={image_diagnostics['dtype']} "
            f"decode_succeeded={image_diagnostics['decode_succeeded']} "
            f"exif_orientation={image_diagnostics['exif_orientation']} "
            f"providers={runtime_diagnostics['providers']} "
            f"detector={runtime_diagnostics['detector_name']} "
            f"model_input={runtime_diagnostics['model_input_shape']} "
            f"det_size={runtime_diagnostics['det_size']} "
            f"det_thresh={runtime_diagnostics['det_thresh']:.3f}"
        )
        if image is None:
            raise ValueError(f"Image {number} is not a valid image")
        faces = safe_face_inference(image)
        print(
            f"[REGISTRATION DETECTION] image={number} face_count={len(faces)} "
            f"faces={_face_detection_summary(faces)}"
        )
        if not faces:
            raise ValueError(
                f"Registration image {number}: no face detected."
            )
        if len(faces) != 1:
            raise ValueError(
                f"Registration image {number}: multiple faces detected."
            )
        face = faces[0]
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
        print(
            f"[REGISTRATION][FACE] image={number} faces_detected={len(faces)} "
            f"bbox=({','.join(f'{value:.1f}' for value in bbox)})"
        )
        _validate_registration_face_quality(face, image, number)
        face_crop, crop_bbox = _create_registration_face_crop(
            face, image, number,
        )
        crop_height, crop_width = face_crop.shape[:2]
        print(
            f"[REGISTRATION][CROP] image={number} "
            f"crop_bbox=({','.join(str(value) for value in crop_bbox)}) "
            f"crop_size={crop_width}x{crop_height} "
            f"margin={REGISTRATION_FACE_CROP_MARGIN}"
        )
        cropped_faces = safe_face_inference(face_crop)
        if not cropped_faces:
            # Some close portraits need the existing registration-only padded
            # detector retry; it still receives only the validated face crop.
            cropped_faces = safe_registration_face_inference(face_crop)
        if not cropped_faces:
            raise ValueError(
                f"Registration image {number}: unable to generate a valid face embedding."
            )
        if len(cropped_faces) != 1:
            raise ValueError(
                f"Registration image {number}: multiple faces detected in face crop."
            )
        try:
            embedding = np.asarray(
                cropped_faces[0].embedding,
                dtype=np.float32,
            ).reshape(-1)
        except Exception as error:
            raise ValueError(
                f"Registration image {number}: unable to generate a valid face embedding."
            ) from error
        if embedding.size != 512 or not np.all(np.isfinite(embedding)):
            raise ValueError(
                f"Registration image {number}: unable to generate a valid face embedding."
            )
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"Registration image {number}: unable to generate a valid face embedding."
            )
        embedding = (embedding / norm).astype(np.float32, copy=False)
        normalized_norm = float(np.linalg.norm(embedding))
        print(
            f"[REGISTRATION][EMBEDDING] image={number} "
            f"original={image.shape[1]}x{image.shape[0]} "
            f"face_bbox=({','.join(f'{value:.1f}' for value in bbox)}) "
            f"crop_bbox=({','.join(str(value) for value in crop_bbox)}) "
            f"crop_size={crop_width}x{crop_height} dimension={embedding.size} "
            f"norm={normalized_norm:.4f} source=face_crop"
        )
        images.append(image)
        crops.append(face_crop)
        embeddings.append(embedding)
    pairwise_similarities = [
        float(np.dot(first, second))
        for index, first in enumerate(embeddings)
        for second in embeddings[index + 1:]
    ]
    if pairwise_similarities and min(pairwise_similarities) >= REGISTRATION_IDENTICAL_SIMILARITY:
        raise ValueError(
            "The five face images are nearly identical. "
            "Please capture the requested different face angles."
        )
    return images, crops, embeddings


def _create_registration_face_crop(face, image, number):
    """Create a bounded face crop without upscaling a small detected face."""
    image_height, image_width = image.shape[:2]
    try:
        x1, y1, x2, y2 = (
            np.asarray(face.bbox, dtype=np.float32).reshape(-1).tolist()
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Registration image {number}: unable to create a valid face crop."
        ) from error

    face_width = x2 - x1
    face_height = y2 - y1
    margin_x = face_width * REGISTRATION_FACE_CROP_MARGIN
    margin_y = face_height * REGISTRATION_FACE_CROP_MARGIN
    crop_x1 = max(0, int(np.floor(x1 - margin_x)))
    crop_y1 = max(0, int(np.floor(y1 - margin_y)))
    crop_x2 = min(image_width, int(np.ceil(x2 + margin_x)))
    crop_y2 = min(image_height, int(np.ceil(y2 + margin_y)))
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    if (
        crop_width < REGISTRATION_FACE_CROP_MIN_SIZE
        or crop_height < REGISTRATION_FACE_CROP_MIN_SIZE
    ):
        raise ValueError(
            f"Registration image {number}: face is too small. Move closer."
        )
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            f"Registration image {number}: unable to create a valid face crop."
        )

    face_crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if not face_crop.size:
        raise ValueError(
            f"Registration image {number}: unable to create a valid face crop."
        )
    if max(crop_width, crop_height) > REGISTRATION_FACE_CROP_MAX_SIZE:
        scale = REGISTRATION_FACE_CROP_MAX_SIZE / max(crop_width, crop_height)
        resized_width = max(1, int(round(crop_width * scale)))
        resized_height = max(1, int(round(crop_height * scale)))
        face_crop = cv2.resize(
            face_crop,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        if (
            resized_width < REGISTRATION_FACE_CROP_MIN_SIZE
            or resized_height < REGISTRATION_FACE_CROP_MIN_SIZE
        ):
            raise ValueError(
                f"Registration image {number}: face is too small. Move closer."
            )
    return face_crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def _validate_registration_face_quality(face, image, number):
    """Reject low-quality registration faces before generating templates."""
    image_height, image_width = image.shape[:2]
    try:
        x1, y1, x2, y2 = (
            np.asarray(face.bbox, dtype=np.float32).reshape(-1).tolist()
        )
    except (TypeError, ValueError):
        print(f"[REGISTRATION FACE] image={number} quality=FAIL reason=invalid_face_box")
        raise ValueError(f"Invalid face position in image {number}")

    face_width = x2 - x1
    face_height = y2 - y1
    face_width_ratio = face_width / max(float(image_width), 1.0)
    face_height_ratio = face_height / max(float(image_height), 1.0)
    det_score = float(getattr(face, "det_score", 0.0))

    if face_width <= 0 or face_height <= 0:
        reason = "invalid_face_box"
    elif face_width < REGISTRATION_MIN_FACE_PIXELS or face_height < REGISTRATION_MIN_FACE_PIXELS:
        reason = "face_too_small"
    elif (
        face_width_ratio < REGISTRATION_MIN_FACE_WIDTH_RATIO
        or face_height_ratio < REGISTRATION_MIN_FACE_HEIGHT_RATIO
    ):
        reason = "face_too_small"
    elif (
        x1 < image_width * REGISTRATION_FACE_EDGE_MARGIN
        or y1 < image_height * REGISTRATION_FACE_EDGE_MARGIN
        or x2 > image_width * (1.0 - REGISTRATION_FACE_EDGE_MARGIN)
        or y2 > image_height * (1.0 - REGISTRATION_FACE_EDGE_MARGIN)
    ):
        reason = "face_partially_outside_frame"
    elif det_score < REGISTRATION_MIN_DET_SCORE:
        reason = "low_detection_confidence"
    else:
        landmarks = np.asarray(getattr(face, "kps", []), dtype=np.float32)
        if landmarks.size and (
            landmarks.shape != (5, 2)
            or not np.all(np.isfinite(landmarks))
            or np.any(landmarks[:, 0] < x1 - face_width * 0.20)
            or np.any(landmarks[:, 0] > x2 + face_width * 0.20)
            or np.any(landmarks[:, 1] < y1 - face_height * 0.20)
            or np.any(landmarks[:, 1] > y2 + face_height * 0.20)
        ):
            reason = "invalid_face_geometry"
        else:
            crop = image[
                max(0, int(y1)):min(image_height, int(np.ceil(y2))),
                max(0, int(x1)):min(image_width, int(np.ceil(x2))),
            ]
            crop = np.asarray(crop, dtype=np.uint8)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
            blur_score = (
                float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if gray is not None and gray.size
                else 0.0
            )
            if blur_score < REGISTRATION_MIN_BLUR_SCORE:
                reason = "face_blurry"
            else:
                reason = None

    blur_score = locals().get("blur_score", 0.0)
    print(
        f"[REGISTRATION FACE] image={number} face_count=1 "
        f"face_width={face_width:.1f} face_height={face_height:.1f} "
        f"face_ratio={face_width_ratio:.4f}x{face_height_ratio:.4f} "
        f"det_score={det_score:.3f} blur_score={blur_score:.1f} "
        f"quality={'FAIL' if reason else 'PASS'}"
        + (f" reason={reason}" if reason else "")
    )
    if reason == "face_too_small":
        raise ValueError(
            f"Registration image {number}: face is too small. Move closer."
        )
    if reason == "face_partially_outside_frame":
        raise ValueError(f"Face is partially outside the frame in image {number}.")
    if reason == "low_detection_confidence":
        raise ValueError(f"Face detection confidence is too low in image {number}.")
    if reason == "face_blurry":
        raise ValueError(
            f"Registration image {number}: face quality is insufficient."
        )
    if reason == "invalid_face_geometry":
        raise ValueError(f"Face geometry is invalid in image {number}.")
    if reason:
        raise ValueError(
            f"Registration image {number}: face quality is insufficient."
        )


def _save_registered_person(gate_no, employee_name, designation, employee_id,
                            images, crops, embeddings):
    """Persist a completely validated registration and update the live cache."""

    registration_id = f"REG_{uuid.uuid4().hex[:12].upper()}"
    storage_key = employee_id or registration_id
    final_dir = KNOWN_FACES_DIR / storage_key
    staging_dir = KNOWN_FACES_DIR / (
        f".{storage_key}.registration-{uuid.uuid4().hex}"
    )
    moved_to_final = False
    conn = None
    try:
        conn = _open_db()
        _ensure_table(conn)
        if employee_id and conn.execute(
            "SELECT 1 FROM registered_persons WHERE employee_id = ?",
            (employee_id,),
        ).fetchone():
            raise ValueError("Employee ID already registered")
        if final_dir.exists():
            raise ValueError("Employee registration folder already exists")

        staging_dir.mkdir(parents=True, exist_ok=False)
        image_paths = []
        crop_paths = []
        crop_dir = staging_dir / "crops"
        if REGISTRATION_SAVE_FACE_CROPS:
            crop_dir.mkdir(parents=True, exist_ok=False)
        for number, image in enumerate(images, start=1):
            image_path = staging_dir / f"image_{number}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Unable to save image {number}")
            image_paths.append(image_path)
            if REGISTRATION_SAVE_FACE_CROPS:
                crop_path = crop_dir / f"image_{number}_face.jpg"
                if not cv2.imwrite(str(crop_path), crops[number - 1]):
                    raise RuntimeError(f"Unable to save face crop {number}")
                crop_paths.append(crop_path)

        # Move only after every image was safely written. The final directory
        # uses a server-generated key, never an Android filename.
        staging_dir.replace(final_dir)
        moved_to_final = True
        final_paths = [final_dir / path.name for path in image_paths]
        fingerprint = _compute_fingerprint(final_paths)
        average_embedding = normalize_embedding(np.mean(
            np.stack(embeddings, axis=0), axis=0,
        ))
        now = time.time()

        conn.execute(
            """
            INSERT INTO registered_persons (
                registration_id, storage_key, gate_no, employee_id,
                employee_name, designation, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                registration_id, storage_key, gate_no, employee_id,
                employee_name, designation, now, now,
            ),
        )
        for number, (path, embedding) in enumerate(
            zip(final_paths, embeddings), start=1,
        ):
            conn.execute(
                """
                INSERT INTO registered_face_embeddings (
                    registration_id, image_number, image_path, embedding,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    registration_id, number, str(path),
                    _embedding_blob(embedding), now,
                ),
            )
        # This is the legacy compatibility record used by the app and DB
        # inspection tooling. The live recognition path must compare against
        # every stored template in registered_face_embeddings.
        conn.execute(
            """
            INSERT INTO known_embeddings (
                person, embedding, image_paths, fingerprint, updated_at
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (person)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                image_paths = EXCLUDED.image_paths,
                fingerprint = EXCLUDED.fingerprint,
                updated_at = EXCLUDED.updated_at
            """,
            (
                storage_key, _embedding_blob(embeddings[0]),
                json.dumps([str(path) for path in final_paths]), fingerprint, now,
            ),
        )
        stored_template_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM registered_face_embeddings
            WHERE registration_id = ?
            """,
            (registration_id,),
        ).fetchone()[0]
        if stored_template_count != 5:
            raise RuntimeError(
                "Registration integrity check failed: expected 5 face templates"
            )
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        if moved_to_final:
            shutil.rmtree(final_dir, ignore_errors=True)
        else:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        if conn is not None:
            conn.close()

    # Keep the compatibility cache entry but do not use it as the primary live
    # recognition template. The matcher compares every stored template globally.
    known_embeddings[storage_key] = [
        normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))
        for embedding in embeddings
    ]
    known_person_metadata[storage_key] = {
        "employee_name": employee_name,
        "person_name": employee_name,
        "employee_id": employee_id,
        "designation": designation,
        "gate_no": gate_no,
    }
    known_face_templates[storage_key] = [
        normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))
        for embedding in embeddings
    ]
    print(
        f"[REGISTER] Person registered successfully: {employee_name} "
        f"storage_key={storage_key} templates={len(known_face_templates[storage_key])}"
    )
    print("[KNOWN CACHE] Reloading registered faces...")
    refresh_known_faces_cache(force=True)
    return registration_id


@app.route("/api/register-person", methods=["POST"])
def api_register_person():
    required_fields = ("gate_no", "employee_name", "designation")
    values = {
        field: request.form.get(field, "").strip()
        for field in required_fields
    }
    for field in required_fields:
        if not values[field]:
            return jsonify({
                "success": False,
                "message": f"{field} is required",
            }), 400
    employee_id = request.form.get("employee_id", "").strip() or None
    try:
        images, crops, embeddings = _validate_registration_images()
        registration_id = _save_registered_person(
            values["gate_no"], values["employee_name"],
            values["designation"], employee_id, images, crops, embeddings,
        )
    except ValueError as error:
        return jsonify({"success": False, "message": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"success": False, "message": str(error)}), 503
    except Exception as error:
        print(f"[ERROR] Person registration failed: {error}")
        return jsonify({
            "success": False,
            "message": "Unable to register employee",
        }), 500
    # Reload the DB-backed cache immediately so the same process can recognize
    # the newly registered employee without waiting for a service restart.
    load_known_faces()

    return jsonify({
        "success": True,
        "message": "Employee registered successfully",
        "registration_id": registration_id,
        "gate_no": values["gate_no"],
        "employee_id": employee_id,
        "employee_name": values["employee_name"],
        "designation": values["designation"],
        "images_registered": 5,
    }), 201


# ============================================================
# ANDROID PERSON UPDATE (PATCH)
# ============================================================

@app.route("/api/update-person/<registration_id>", methods=["PATCH"])
def api_update_person(registration_id):
    """Update one or more fields of a registered person.

    All fields are optional — only the supplied fields are updated.
    Accepted form/JSON fields: gate_no, employee_name, designation, employee_id.
    """

    # Accept both form-data and JSON payloads.
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict()

    updatable_fields = ("gate_no", "employee_name", "designation", "employee_id")
    updates = {}
    for field in updatable_fields:
        if field in payload:
            value = str(payload[field]).strip()
            # Allow explicitly clearing employee_id by sending an empty string.
            if field == "employee_id":
                updates[field] = value or None
            else:
                if not value:
                    return jsonify({
                        "success": False,
                        "message": f"{field} cannot be empty",
                    }), 400
                updates[field] = value

    if not updates:
        return jsonify({
            "success": False,
            "message": "No updatable fields provided. "
                       "Accepted fields: gate_no, employee_name, designation, employee_id",
        }), 400

    conn = None
    try:
        conn = _open_db()
        _ensure_table(conn)

        # Fetch the existing record.
        row = conn.execute(
            "SELECT storage_key, gate_no, employee_id, employee_name, designation "
            "FROM registered_persons WHERE registration_id = ?",
            (registration_id,),
        ).fetchone()
        if row is None:
            return jsonify({
                "success": False,
                "message": "Registration ID not found",
            }), 404

        old_storage_key, old_gate_no, old_employee_id, old_employee_name, old_designation = row

        # If the caller wants to change employee_id, ensure uniqueness.
        new_employee_id = updates.get("employee_id", old_employee_id)
        if (
            "employee_id" in updates
            and new_employee_id is not None
            and new_employee_id != old_employee_id
        ):
            conflict = conn.execute(
                "SELECT 1 FROM registered_persons WHERE employee_id = ? AND registration_id != ?",
                (new_employee_id, registration_id),
            ).fetchone()
            if conflict:
                return jsonify({
                    "success": False,
                    "message": "Employee ID already registered to another person",
                }), 400

        # Build the dynamic UPDATE.
        set_clauses = []
        params = []
        for field, value in updates.items():
            set_clauses.append(f"{field} = ?")
            params.append(value)
        set_clauses.append("updated_at = ?")
        params.append(time.time())
        params.append(registration_id)

        conn.execute(
            f"UPDATE registered_persons SET {', '.join(set_clauses)} "
            f"WHERE registration_id = ?",
            tuple(params),
        )
        conn.commit()
    except Exception as error:
        if conn is not None:
            conn.rollback()
        print(f"[ERROR] Person update failed: {error}")
        return jsonify({
            "success": False,
            "message": "Unable to update employee",
        }), 500
    finally:
        if conn is not None:
            conn.close()

    # Refresh the live metadata cache so recognition labels update immediately.
    final_values = {
        "gate_no": updates.get("gate_no", old_gate_no),
        "employee_name": updates.get("employee_name", old_employee_name),
        "employee_id": updates.get("employee_id", old_employee_id),
        "designation": updates.get("designation", old_designation),
    }
    if old_storage_key in known_person_metadata:
        known_person_metadata[old_storage_key] = {
            **known_person_metadata[old_storage_key],
            "gate_no": final_values["gate_no"],
            "employee_name": final_values["employee_name"],
            "person_name": final_values["employee_name"],
            "employee_id": final_values["employee_id"],
            "designation": final_values["designation"],
        }

    # Keep the live recognition cache in sync with the persisted DB after edits.
    refresh_known_faces_cache(force=True)

    return jsonify({
        "success": True,
        "message": "Employee updated successfully",
        "registration_id": registration_id,
        **final_values,
    }), 200


# ============================================================
# ANDROID PERSON DELETE
# ============================================================

@app.route("/api/delete-person/<registration_id>", methods=["DELETE"])
def api_delete_person(registration_id):
    """Delete a registered person and remove them from live recognition."""

    conn = None
    try:
        conn = _open_db()
        _ensure_table(conn)

        row = conn.execute(
            "SELECT storage_key FROM registered_persons WHERE registration_id = ?",
            (registration_id,),
        ).fetchone()
        if row is None:
            return jsonify({
                "success": False,
                "message": "Registration ID not found",
            }), 404

        storage_key = row[0]

        # Remove from all three database tables.
        conn.execute(
            "DELETE FROM registered_face_embeddings WHERE registration_id = ?",
            (registration_id,),
        )
        conn.execute(
            "DELETE FROM registered_persons WHERE registration_id = ?",
            (registration_id,),
        )
        conn.execute(
            "DELETE FROM known_embeddings WHERE person = ?",
            (storage_key,),
        )
        conn.commit()

        # Remove the image folder from disk.
        person_dir = KNOWN_FACES_DIR / storage_key
        if person_dir.exists():
            shutil.rmtree(person_dir, ignore_errors=True)

    except Exception as error:
        if conn is not None:
            conn.rollback()
        print(f"[ERROR] Person deletion failed: {error}")
        return jsonify({
            "success": False,
            "message": "Unable to delete employee",
        }), 500
    finally:
        if conn is not None:
            conn.close()

    # Evict from the live recognition caches so the person
    # stops being recognized immediately.
    known_embeddings.pop(storage_key, None)
    known_person_metadata.pop(storage_key, None)
    known_face_templates.pop(storage_key, None)
    refresh_known_faces_cache(force=True)

    return jsonify({
        "success": True,
        "message": "Employee deleted successfully",
        "registration_id": registration_id,
    }), 200


# ============================================================
# ADMIN / USER AUTH
# ============================================================

def _auth_login_response(role, account_row, fcm_token):

    account_id = int(account_row[0])
    username = account_row[2]
    _upsert_fcm_token(account_id, role, fcm_token, is_active=True)
    payload = {
        "success": True,
        "message": "Login successful",
        "role": role,
        "username": username,
        "account_id": account_id,
        "fcm_token": fcm_token,
    }
    if role == "admin":
        payload["admin_id"] = account_id
    else:
        payload["user_id"] = account_id
    return jsonify(payload), 200


@app.route("/api/admin/login", methods=["POST"])
def api_admin_login():

    payload = _json_or_form_payload()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    fcm_token = str(payload.get("fcm_token", "")).strip()

    if not username or not password:
        return jsonify({
            "success": False,
            "message": "username and password are required",
        }), 400
    if not fcm_token:
        return jsonify({
            "success": False,
            "message": "fcm_token is required",
        }), 400

    account = _auth_account_by_credentials("admin", username, password)
    if account is None:
        return jsonify({
            "success": False,
            "message": "Invalid admin credentials",
        }), 401

    return _auth_login_response("admin", account, fcm_token)


@app.route("/api/user/login", methods=["POST"])
def api_user_login():

    payload = _json_or_form_payload()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", "")).strip()
    fcm_token = str(payload.get("fcm_token", "")).strip()

    if not username or not password:
        return jsonify({
            "success": False,
            "message": "username and password are required",
        }), 400
    if not fcm_token:
        return jsonify({
            "success": False,
            "message": "fcm_token is required",
        }), 400

    account = _auth_account_by_credentials("user", username, password)
    if account is None:
        return jsonify({
            "success": False,
            "message": "Invalid user credentials",
        }), 401

    return _auth_login_response("user", account, fcm_token)


def _auth_logout(role, id_field, account_id):

    payload = _json_or_form_payload()
    fcm_token = str(payload.get("fcm_token", "")).strip()
    if not fcm_token:
        return jsonify({
            "success": False,
            "message": "fcm_token is required",
        }), 400

    try:
        account_id = int(account_id)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": f"{id_field} must be a valid integer",
        }), 400

    account = _auth_account_by_id(role, account_id)
    if account is None:
        return jsonify({
            "success": False,
            "message": f"{id_field} not found",
        }), 404

    updated = _set_fcm_token_active_status(account_id, role, fcm_token, False)
    if not updated:
        return jsonify({
            "success": False,
            "message": "Matching active device token not found",
        }), 404

    return jsonify({
        "success": True,
        "message": "Logout successful",
        "role": role,
        id_field: account_id,
        "fcm_token": fcm_token,
        "is_active": False,
    }), 200


@app.route("/api/admin/logout", methods=["POST"])
def api_admin_logout():

    payload = _json_or_form_payload()
    return _auth_logout(
        "admin",
        "admin_id",
        payload.get("admin_id"),
    )


@app.route("/api/user/logout", methods=["POST"])
def api_user_logout():

    payload = _json_or_form_payload()
    return _auth_logout(
        "user",
        "user_id",
        payload.get("user_id"),
    )


# ============================================================
# AUTHENTICATED CAMERA DASHBOARD
# ============================================================

@app.route("/dashboard/login", methods=["GET", "POST"])
def dashboard_login():
    if request.method == "GET":
        if not app.secret_key:
            return render_template(
                "dashboard_login.html",
                error="Dashboard login is not configured on this server.",
            ), 503
        if session.get("dashboard_user"):
            return redirect(url_for("camera_dashboard"))
        return render_template("dashboard_login.html", error=None)

    if not app.secret_key:
        app.logger.error("Dashboard login is unavailable: FLASK_SECRET_KEY is not configured")
        return render_template(
            "dashboard_login.html",
            error="Dashboard login is not configured on this server.",
        ), 503

    role = _normalize_role(request.form.get("role", "user"))
    username = str(request.form.get("username", "")).strip()
    password = str(request.form.get("password", ""))
    if not role or not username or not password:
        return render_template(
            "dashboard_login.html",
            error="Enter a valid role, username, and password.",
        ), 400

    account = _auth_account_by_credentials(role, username, password)
    if account is None:
        return render_template(
            "dashboard_login.html",
            error="Invalid username or password.",
        ), 401
    session.clear()
    session["dashboard_user"] = {"username": account[2], "role": role}
    return redirect(url_for("camera_dashboard"))


@app.route("/dashboard/logout", methods=["POST"])
def dashboard_logout():
    if not app.secret_key:
        return redirect(url_for("dashboard_login"))
    session.clear()
    return redirect(url_for("dashboard_login"))


@app.route("/")
def dashboard_root():
    return redirect(url_for("camera_dashboard"))


@app.route("/dashboard")
def camera_dashboard():
    if not app.secret_key:
        return render_template(
            "dashboard_login.html",
            error="Dashboard login is not configured on this server.",
        ), 503
    dashboard_user = session.get("dashboard_user")
    if not dashboard_user:
        return redirect(url_for("dashboard_login"))
    return render_template("dashboard.html", dashboard_user=dashboard_user)


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    if not app.secret_key:
        return jsonify({"success": False, "message": "Dashboard authentication is not configured"}), 503
    if not session.get("dashboard_user"):
        return jsonify({"success": False, "message": "Authentication required"}), 401

    selected_id = next(
        (known_id for known_id in camera_states if known_id.lower() == camera_id.lower()),
        None,
    )
    if selected_id is None:
        return jsonify({"success": False, "message": "Camera not found"}), 404
    state = camera_states[selected_id]

    def stream_frames():
        last_frame_counter = -1
        while not shutdown_event.is_set():
            with state["lock"]:
                frame = state["latest_annotated_frame"]
                frame_counter = state["frame_counter"]
            if frame is None or frame_counter == last_frame_counter:
                shutdown_event.wait(0.1)
                continue
            last_frame_counter = frame_counter
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return Response(
        stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# SERVE ALERT IMAGES
# ============================================================

@app.route(
    "/alerts/<filename>",
    methods=["GET"]
)
def alert_image(filename):

    return send_from_directory(
        ALERT_IMAGE_DIR,
        filename
    )


@app.route(
    "/known-faces/<storage_key>/<filename>",
    methods=["GET"]
)
def known_face_image(storage_key, filename):

    return send_from_directory(
        KNOWN_FACES_DIR / storage_key,
        filename
    )


# ============================================================
# REST API
# ============================================================

def _camera_api_status(camera):
    if not camera.get("enabled", False):
        return "disabled"
    snapshot = _camera_snapshot(camera["camera_id"])
    raw_status = snapshot["status"] if snapshot else "Starting..."
    if raw_status == "Starting...":
        return "connecting"
    if raw_status.startswith("Error:"):
        return "reconnecting"
    if raw_status.startswith("Stream lost") or raw_status.startswith("RTSP connection"):
        return "reconnecting"
    if raw_status == "LIVE" or (snapshot and snapshot["connected"]):
        return "online"
    return "offline"


def _safe_camera_status_message(status):
    message = str(status or "Unknown camera status")
    password = os.getenv("ALCON_PASSWORD", "")
    if password:
        message = message.replace(password, "[redacted]")
        message = message.replace(quote(password, safe=""), "[redacted]")
    return re.sub(
        r"rtsps?://[^\s\"'<>]+",
        lambda match: redact_rtsp_url(match.group(0)),
        message,
        flags=re.IGNORECASE,
    )[:300]


def _camera_api_payload(camera):
    snapshot = _camera_snapshot(camera["camera_id"])
    return {
        "camera_id": camera["camera_id"],
        "name": camera["name"],
        "kpi": camera.get("kpi", "person"),
        "channel": camera.get("channel"),
        "status": _camera_api_status(camera),
        "status_message": _safe_camera_status_message(
            snapshot["status"] if snapshot else "Starting..."
        ),
        "enabled": bool(camera.get("enabled", False)),
    }


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    return jsonify({
        "success": True,
        "cameras": [
            _camera_api_payload(camera)
            for camera in CAMERAS
        ],
    })


@app.route("/api/cameras/<camera_id>", methods=["GET"])
def api_camera(camera_id):
    camera = next(
        (
            item for item in CAMERAS
            if item["camera_id"].lower() == camera_id.lower()
        ),
        None,
    )
    if camera is None:
        return jsonify({
            "success": False,
            "message": "Camera not found",
        }), 404
    return jsonify({
        "success": True,
        "camera": _camera_api_payload(camera),
    })

@app.route(
    "/api/status",
    methods=["GET"]
)
def api_status():

    camera_id = request.args.get("camera_id", "").strip()
    if not camera_id:
        camera_id = request.args.get("channel", "").strip()
    selected = _camera_by_id_or_channel(camera_id=camera_id if camera_id and not camera_id.isdigit() else None, channel=camera_id if camera_id and camera_id.isdigit() else request.args.get("channel", ""))
    snapshot = _camera_snapshot(selected["camera_id"])
    if snapshot is None:
        return jsonify({
            "status": "Unknown camera",
            "connected_clients": connected_clients,
            "known_faces_count": len(known_embeddings),
        }), 404

    return jsonify(
        {
            "camera_id": snapshot["camera_id"],
            "camera_name": snapshot["camera_name"],
            "status": snapshot["status"],
            "frame_available": snapshot["frame"] is not None,
            "frame_bytes": len(snapshot["frame"] or b""),
            "connected_clients":
                connected_clients,
            "known_faces_count":
                len(known_embeddings),
        }
    )


@app.route(
    "/api/mobile",
    methods=["GET"]
)
def api_mobile():
    """Return the stable, combined response used by the Android app."""
    limit = request.args.get(
        "limit",
        # No limit means the Android login receives the complete persisted
        # detection-event history, not only the current phone's cache.
        default=0,
        type=int
    )
    limit = max(0, limit)
    detection_type = request.args.get("type", default="").strip().lower()
    valid_detection_types = {
        "known_person",
        "unknown_person",
        "vehicle",
        "fall_detected",
        "smoke_detected",
        "fire_detected",
        "object_theft",
    }
    if detection_type not in valid_detection_types:
        detection_type = ""
    camera_id = request.args.get("camera_id", default="").strip() or ""
    include_snapshot = request.args.get(
        "include_snapshot",
        default="true"
    ).lower() in {"1", "true", "yes"}

    # Read the persisted master history first so summary totals remain correct
    # for every mobile installation. Filtering changes only the returned list.
    all_detections = detection_database.latest(0, camera_id=camera_id or None)
    detections = [
        item
        for item in all_detections
        if not detection_type or item["type"] == detection_type
    ]
    if limit:
        detections = detections[:limit]
    detection_summary = {
        "total_events": len(all_detections),
        "known_events": sum(
            item["type"] == "known_person"
            for item in all_detections
        ),
        "unknown_events": sum(
            item["type"] == "unknown_person"
            for item in all_detections
        ),
        "unknown_people": sum(
            int(item.get("unknown_count") or 1)
            for item in all_detections
            if item["type"] == "unknown_person"
        ),
        "vehicle_events": sum(
            item["type"] == "vehicle"
            for item in all_detections
        ),
        "fall_events": sum(
            item["type"] == "fall_detected"
            for item in all_detections
        ),
        "smoke_events": sum(
            item["type"] == "smoke_detected"
            for item in all_detections
        ),
        "fire_events": sum(
            item["type"] == "fire_detected"
            for item in all_detections
        ),
        "object_theft_events": sum(
            item["type"] == "object_theft"
            for item in all_detections
        ),
    }
    alert_items = [
        item
        for item in alert_history
        if not camera_id or item.get("camera_id") == camera_id
    ]
    alert_limit = len(alert_items) if limit == 0 else min(limit, len(alert_items))
    alerts = alert_items[-alert_limit:]
    alerts.reverse()

    gates = sorted({
        item["gate"]
        for item in detections
        if item.get("gate")
    } | {camera_states[camera_id]["camera_name"] if camera_id in camera_states else _default_camera()["name"]})

    cameras_payload = []
    for camera in _enabled_cameras():
        snapshot = _camera_snapshot(camera["camera_id"])
        cameras_payload.append({
            "camera_id": camera["camera_id"],
            "camera_name": camera["name"],
            "status": snapshot["status"] if snapshot else "Starting...",
        })

    snapshot_state = _camera_snapshot(camera_id or None)
    snapshot = None
    if include_snapshot:
        frame = snapshot_state["frame"] if snapshot_state else None
        if frame is not None:
            snapshot = {
                "mime_type": "image/jpeg",
                "base64": base64.b64encode(frame).decode("utf-8"),
            }

    return jsonify(
        {
            "success": True,
            "version": 1,
            "data": {
                "status": snapshot_state["status"] if snapshot_state else "Unknown",
                "connected_clients": connected_clients,
                "known_faces_count": len(known_embeddings),
                "people": sorted(known_embeddings.keys()),
                "alerts": alerts,
                "detections": detections,
                "detection_summary": detection_summary,
                "snapshot": snapshot,
                "camera_id": camera_id or _default_camera()["camera_id"],
                "camera_name": snapshot_state["camera_name"] if snapshot_state else _default_camera()["name"],
                "cameras": cameras_payload,
            },
            "options": {
                "gates": gates,
                "people": sorted(known_embeddings.keys()),
                "detection_types": [
                    "known_person",
                    "unknown_person",
                    "vehicle",
                    "fall_detected",
                    "smoke_detected",
                    "fire_detected",
                    "object_theft",
                ],
                "vehicle_types": [
                    "two_wheeler",
                    "four_wheeler",
                ],
            },
        }
    )


@app.route(
    "/api/known-faces",
    methods=["GET"]
)
def api_known_faces():

    return jsonify(
        {
            "people":
                sorted(
                    known_embeddings.keys()
                )
        }
    )


@app.route(
    "/api/registered-employees",
    methods=["GET"]
)
def api_registered_employees():

    employees = _registered_employees_payload()

    return jsonify(
        {
            "success": True,
            "count": len(employees),
            "employees": employees,
        }
    )


@app.route(
    "/api/registered-employees/<registration_id>",
    methods=["GET"]
)
def api_registered_employee(registration_id):

    employee = _registered_employee_payload(registration_id)

    if employee is None:
        return jsonify(
            {
                "success": False,
                "message": "Registration ID not found",
            }
        ), 404

    return jsonify(
        {
            "success": True,
            "employee": employee,
        }
    )


@app.route(
    "/api/alerts",
    methods=["GET"]
)
def api_alerts():

    limit = request.args.get(
        "limit",
        default=20,
        type=int
    )
    camera_id = request.args.get("camera_id", "").strip() or ""

    items = [
        item
        for item in alert_history
        if not camera_id or item.get("camera_id") == camera_id
    ][-limit:]

    items.reverse()

    return jsonify(
        {
            "alerts": items
        }
    )


@app.route(
    "/api/detections",
    methods=["GET"]
)
def api_detections():

    limit = request.args.get(
        "limit",
        default=0,
        type=int
    )
    camera_id = request.args.get("camera_id", "").strip() or None

    return jsonify(
        {
            "detections": detection_database.latest(limit, camera_id=camera_id)
        }
    )


@app.route(
    "/api/snapshot",
    methods=["GET"]
)
def api_snapshot():
    camera_id = request.args.get("camera_id", "").strip()
    channel = request.args.get("channel", "").strip()
    selected = _camera_by_id_or_channel(camera_id=camera_id or None, channel=channel or None)
    snapshot = _camera_snapshot(selected["camera_id"])
    if snapshot is None:
        return jsonify({"error": "Unknown camera"}), 404

    frame = snapshot["frame"]


    if frame is None:

        return jsonify(
            {
                "error":
                    "No frame available yet"
            }
        ), 503


    b64_frame = base64.b64encode(
        frame
    ).decode("utf-8")


    return jsonify(
        {
            "image": b64_frame,
            "status": snapshot["status"],
            "camera_id": snapshot["camera_id"],
            "camera_name": snapshot["camera_name"],
        }
    )


# ============================================================
# MANUAL TEST NOTIFICATION
# ============================================================

@app.route(
    "/api/test-notification",
    methods=["POST"]
)
def test_notification():
    default_camera = _default_camera()
    snapshot = _camera_snapshot(default_camera["camera_id"])
    frame = snapshot["frame"] if snapshot else None


    if frame is None:

        return jsonify(
            {
                "success": False,
                "error":
                    "No camera frame available"
            }
        ), 503


    queued = push_alert(
        "New person detected "
        f"on {default_camera['name']}",
        frame,
        default_camera["name"],
        camera_id=default_camera["camera_id"],
        camera_name=default_camera["name"],
    )

    if not queued:
        return jsonify({
            "success": False,
            "error": "Email alert was not queued; check email configuration and worker status.",
        }), 503

    return jsonify(
        {
            "success": True,
            "message":
                "Test email alert queued"
        }
    )


# ============================================================
# SOCKET.IO EVENTS
# ============================================================

@socketio.on("connect")
def on_connect():

    global connected_clients

    connected_clients += 1

    print(
        "[INFO] Client connected. "
        f"Total: {connected_clients}"
    )


@socketio.on("disconnect")
def on_disconnect():

    global connected_clients

    connected_clients = max(
        0,
        connected_clients - 1
    )

    print(
        "[INFO] Client disconnected. "
        f"Total: {connected_clients}"
    )


@socketio.on("join_camera")
def join_camera(payload):

    camera_id = str((payload or {}).get("camera_id", "")).strip()
    if not camera_id or camera_id not in camera_states:
        return {"success": False, "message": "camera_id is required"}
    join_room(_camera_room(camera_id))
    return {
        "success": True,
        "camera_id": camera_id,
        "room": _camera_room(camera_id),
    }


@socketio.on("leave_camera")
def leave_camera(payload):

    camera_id = str((payload or {}).get("camera_id", "")).strip()
    if not camera_id:
        return {"success": False, "message": "camera_id is required"}
    leave_room(_camera_room(camera_id))
    return {
        "success": True,
        "camera_id": camera_id,
        "room": _camera_room(camera_id),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    global camera_manager
    global shutdown_started

    print("=" * 60)

    print(
        " ALCON MULTI-CAMERA "
        "FACE RECOGNITION"
    )

    print(" EMAIL ALERTS AND HTTP CAMERA DASHBOARD")

    print("=" * 60)


    # ----------------------------------------
    # Email delivery is independent of camera inference and Firebase.
    # ----------------------------------------

    initialize_email_alerts()
    initialize_camera_states()
    print(
        "[INFO][FACE GATES] "
        f"known_threshold={RECOGNITION_THRESHOLD:.2f} "
        f"candidate_threshold={RECOGNITION_CANDIDATE_THRESHOLD:.2f} "
        f"margin={RECOGNITION_MARGIN:.2f} "
        f"min_face_score={FACE_RECOGNITION_MIN_DET_SCORE:.2f} "
        f"unknown_face_score={UNKNOWN_FACE_MIN_SCORE:.2f}"
    )


    # ----------------------------------------
    # Face model
    # ----------------------------------------

    for camera_config in _enabled_cameras():
        configure_h264_stream(camera_config)

    initialize_face_model()
    initialize_fall_detector()
    initialize_ppe_detector()
    initialize_smoke_fire_detector()
    initialize_object_theft_detector()
    initialize_detection_events()


    # ----------------------------------------
    # Known faces
    # ----------------------------------------

    load_known_faces()
    initialize_auth_accounts()


    # ----------------------------------------
    # Camera worker
    # ----------------------------------------
    camera_manager = CameraManager(
        _enabled_cameras(),
        camera_worker,
        shutdown_event=shutdown_event,
        release_resources=_release_camera_capture,
        rtsp_url_builder=build_rtsp_url,
        frame_store=camera_frame_store,
    )
    camera_manager.start()

    def _shutdown_handler(signum, frame):
        global shutdown_started
        if shutdown_started:
            return
        shutdown_started = True
        print(f"[INFO] Shutdown signal received: {signum}")
        shutdown_event.set()
        print("[INFO] Stopping camera workers...")
        camera_manager.stop_all(timeout=5)
        # Returning from a custom signal handler would leave socketio.run()
        # serving. Let main() unwind it through the normal exception path.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)


    print()

    print(
        f"[INFO] REST API base: "
        f"http://<server-ip>:{PORT}/api/"
    )

    print(
        f"[INFO] Alert images: "
        f"{PUBLIC_BASE_URL}/alerts/"
    )

    print("[INFO] Dashboard: /dashboard")
    print("[INFO] Detection alert delivery: configured email recipients")

    print(
        "[INFO] Socket.IO events: "
        "'face_alert', 'detection_event' (no video frames)"
    )

    print(
        "[INFO] Enabled cameras: "
        + ", ".join(
            f"{camera['camera_id']}={camera['name']}"
            for camera in _enabled_cameras()
        )
    )

    print(
        "[INFO] Press CTRL+C "
        "to stop."
    )

    print()


    try:
        socketio.run(
            app,
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        pass

    if not shutdown_started:
        shutdown_started = True
        shutdown_event.set()
        print("[INFO] Stopping camera workers...")
        camera_manager.stop_all(timeout=5)
    if email_alert_queue is not None:
        email_alert_queue.stop(timeout=20)
    print("[INFO] Flask/Socket.IO shutdown complete")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
