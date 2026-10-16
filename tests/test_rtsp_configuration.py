import json
from urllib.parse import urlsplit

import pytest

from camera import rtsp_url
from config.cameras import load_cameras


def test_subtype_one_builds_main_stream_and_subtype_two_builds_substream(
    monkeypatch,
):
    monkeypatch.setenv("ALCON_PASSWORD", "secret")
    camera = {"nvr_ip": "192.168.191.40", "channel": 1}

    main_url = rtsp_url.build_rtsp_url({**camera, "subtype": 1})
    substream_url = rtsp_url.build_rtsp_url({**camera, "subtype": 2})

    assert urlsplit(main_url).path == "/Streaming/Channels/101"
    assert urlsplit(substream_url).path == "/Streaming/Channels/102"
    assert urlsplit(main_url).password == "secret"


def test_configured_endpoint_wins_but_uses_environment_credentials(monkeypatch):
    monkeypatch.setenv("ALCON_PASSWORD", "secret")
    url = rtsp_url.build_rtsp_url(
        {
            "RTSP_URL": (
                "rtsp://legacy:embedded-placeholder@nvr.local:8554"
                "/Streaming/channels/101?profile=main"
            ),
            "nvr_ip": "192.168.191.40",
            "channel": 1,
            "subtype": 2,
        }
    )
    parsed = urlsplit(url)

    assert parsed.hostname == "nvr.local"
    assert parsed.port == 8554
    assert parsed.path == "/Streaming/channels/101"
    assert parsed.query == "profile=main"
    assert parsed.username == "admin"
    assert parsed.password == "secret"


def test_camera_loader_strips_credentials_from_configured_endpoint(tmp_path):
    config_file = tmp_path / "cameras.json"
    config_file.write_text(
        json.dumps(
            [
                {
                    "camera_id": "CAM001",
                    "name": "Office",
                    "nvr_ip": "192.168.191.40",
                    "channel": 1,
                    "subtype": 1,
                    "enabled": True,
                    "RTSP_URL": (
                        "rtsp://admin:embedded-placeholder@192.168.191.40"
                        ":554/Streaming/Channels/101"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )

    camera = load_cameras(config_file)[0]
    endpoint = urlsplit(camera["rtsp_url"])

    assert endpoint.hostname == "192.168.191.40"
    assert endpoint.path == "/Streaming/Channels/101"
    assert endpoint.username is None
    assert endpoint.password is None


def test_cam001_uses_known_unknown_kpi():
    camera = next(
        camera for camera in load_cameras()
        if camera["camera_id"] == "CAM001"
    )

    assert camera["kpi"] == "known_unknown"


def test_rtsp_log_redaction_removes_credentials_and_query():
    redacted = rtsp_url.redact_rtsp_url(
        "rtsp://admin:secret@192.168.191.40:554/Streaming/Channels/101"
        "?token=private"
    )

    assert redacted == (
        "rtsp://***@192.168.191.40:554/Streaming/Channels/101"
    )
    assert "secret" not in redacted
    assert "private" not in redacted


def test_missing_password_is_reported(monkeypatch):
    monkeypatch.delenv("ALCON_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="ALCON_PASSWORD"):
        rtsp_url.build_rtsp_url({"nvr_ip": "192.168.191.40"})
