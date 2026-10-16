"""RTSP endpoint construction and credential-safe logging helpers."""

import os
from urllib.parse import quote, urlsplit, urlunsplit


RTSP_PORT = 554


def build_rtsp_url(camera_config):
    """Build one authenticated RTSP URL from config and environment settings."""
    rtsp_password = os.getenv("ALCON_PASSWORD", "")
    if not rtsp_password:
        raise RuntimeError("ALCON_PASSWORD environment variable is not set.")

    username = quote("admin", safe="")
    password = quote(rtsp_password, safe="")
    configured_url = (
        camera_config.get("RTSP_URL")
        or camera_config.get("rtsp_url")
    )

    if configured_url:
        try:
            endpoint = urlsplit(str(configured_url).strip())
            port = endpoint.port or RTSP_PORT
        except ValueError as exc:
            raise ValueError("Invalid configured RTSP_URL endpoint.") from exc
        if (
            endpoint.scheme.lower() not in {"rtsp", "rtsps"}
            or not endpoint.hostname
            or not endpoint.path
        ):
            raise ValueError("Invalid configured RTSP_URL endpoint.")
        host = endpoint.hostname
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{username}:{password}@{host}:{port}"
        return urlunsplit(
            (endpoint.scheme, netloc, endpoint.path, endpoint.query, "")
        )

    channel = int(camera_config.get("channel", 1))
    subtype = int(camera_config.get("subtype", 1))
    if subtype not in {1, 2}:
        raise ValueError(
            f"Invalid RTSP subtype {subtype!r}; expected 1 (main) or 2 (substream)."
        )
    nvr_ip = str(camera_config.get("nvr_ip") or os.getenv("NVR_IP", "")).strip()
    if not nvr_ip:
        raise ValueError("NVR IP is not configured.")

    host = f"[{nvr_ip}]" if ":" in nvr_ip else nvr_ip
    channel_path = channel * 100 + subtype
    return (
        f"rtsp://{username}:{password}@{host}:{RTSP_PORT}"
        f"/Streaming/Channels/{channel_path}"
    )


def redact_rtsp_url(rtsp_url):
    """Return an RTSP endpoint suitable for logs, without credentials/query."""
    try:
        endpoint = urlsplit(str(rtsp_url))
        if not endpoint.scheme or not endpoint.hostname:
            return "<invalid RTSP URL>"
        host = endpoint.hostname
        if ":" in host:
            host = f"[{host}]"
        port = f":{endpoint.port}" if endpoint.port is not None else ""
        auth = "***@" if endpoint.username is not None or endpoint.password is not None else ""
        return urlunsplit(
            (endpoint.scheme, f"{auth}{host}{port}", endpoint.path, "", "")
        )
    except ValueError:
        return "<invalid RTSP URL>"
