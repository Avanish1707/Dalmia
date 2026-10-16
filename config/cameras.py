"""Camera configuration loading and lookup helpers."""

import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from config.settings import PROJECT_ROOT

DEFAULT_NVR_IP = os.getenv("NVR_IP", "192.168.191.40")
CAMERA_CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.json"


def _endpoint_without_credentials(rtsp_url, camera_id):
    if not isinstance(rtsp_url, str) or not rtsp_url.strip():
        raise ValueError(f"Invalid RTSP_URL for {camera_id!r}")
    try:
        endpoint = urlsplit(rtsp_url.strip())
        port = endpoint.port
    except ValueError as exc:
        raise ValueError(f"Invalid RTSP_URL for {camera_id!r}") from exc
    if (
        endpoint.scheme.lower() not in {"rtsp", "rtsps"}
        or not endpoint.hostname
        or not endpoint.path
    ):
        raise ValueError(f"Invalid RTSP_URL for {camera_id!r}")

    host = endpoint.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit(
        (endpoint.scheme, netloc, endpoint.path, endpoint.query, "")
    )


def load_cameras(config_path=CAMERA_CONFIG_PATH):
    """Load camera definitions while preserving environment overrides."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        configured = json.load(config_file)

    cameras = []
    seen_camera_ids = set()
    seen_channels = set()

    for item in configured:
        camera_id = str(item.get("camera_id", "")).strip()
        if not camera_id:
            raise ValueError("Camera entry is missing camera_id")
        if camera_id in seen_camera_ids:
            raise ValueError(f"Duplicate camera_id: {camera_id!r}")
        seen_camera_ids.add(camera_id)

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Camera entry is missing name for {camera_id!r}")

        nvr_ip = os.getenv(f"{camera_id}_NVR_IP", item.get("nvr_ip", DEFAULT_NVR_IP))
        channel = int(os.getenv(f"{camera_id}_CHANNEL", str(item.get("channel", 1))))
        if channel in seen_channels:
            raise ValueError(f"Duplicate channel value: {channel} used by {camera_id!r}")
        seen_channels.add(channel)

        subtype = int(os.getenv(f"{camera_id}_SUBTYPE", str(item.get("subtype", 1))))
        if subtype not in {1, 2}:
            raise ValueError(f"Invalid subtype {subtype!r} for {camera_id!r}; expected 1 or 2")

        enabled_value = os.getenv(
            f"{camera_id}_ENABLED",
            str(bool(item.get("enabled", False))),
        )
        enabled = enabled_value.strip().lower() not in {"false", "0", "no", "off", ""}
        if not isinstance(item.get("enabled", enabled), bool):
            # Keep the configuration contract strict while preserving env overrides.
            if not isinstance(item.get("enabled", True), bool):
                raise ValueError(f"Invalid enabled flag for {camera_id!r}")

        camera = {
            "camera_id": camera_id,
            "name": name,
            "ip": item.get("ip"),
            "nvr_ip": nvr_ip,
            "kpi": str(item.get("kpi", "person")).strip().lower(),
            "enabled": enabled,
            "channel": channel,
            "subtype": subtype,
        }
        configured_rtsp_url = item.get("RTSP_URL") or item.get("rtsp_url")
        if configured_rtsp_url is not None:
            camera["rtsp_url"] = _endpoint_without_credentials(
                configured_rtsp_url,
                camera_id,
            )
        cameras.append(camera)

    enabled_override = os.getenv("ENABLED_CAMERAS", "").strip()
    if enabled_override:
        selected = {
            value.strip().upper()
            for value in enabled_override.split(",")
            if value.strip()
        }
        if selected != {"ALL"}:
            known_ids = {camera["camera_id"] for camera in cameras}
            unknown = selected - known_ids
            if unknown:
                raise ValueError(
                    "Unknown camera IDs in ENABLED_CAMERAS: "
                    + ", ".join(sorted(unknown))
                )
            for camera in cameras:
                camera["enabled"] = camera["camera_id"] in selected

    return cameras


CAMERAS = load_cameras()


def enabled_cameras():
    return [camera for camera in CAMERAS if camera.get("enabled", False)]


def camera_by_id(camera_id):
    if camera_id:
        normalized = str(camera_id).strip().lower()
        return next(
            (camera for camera in CAMERAS if camera["camera_id"].lower() == normalized),
            None,
        )
    return None


__all__ = ["CAMERAS", "CAMERA_CONFIG_PATH", "camera_by_id", "enabled_cameras", "load_cameras"]
