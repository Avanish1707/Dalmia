import importlib
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture(scope="module")
def vms_module():
    monkeypatch = pytest.MonkeyPatch()
    import database
    import detection_database

    class FakeDetectionDatabase:
        def __init__(self):
            pass

        def save(self, event):
            return 1

        def latest(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(database, "validate_postgres_connection", lambda: True)
    monkeypatch.setattr(detection_database, "DetectionDatabase", FakeDetectionDatabase)
    module = importlib.import_module("alcon_stream_new")
    yield module
    monkeypatch.undo()


def test_email_test_routes_are_registered_on_flask_app(vms_module):
    rules = {
        (rule.rule, method)
        for rule in vms_module.app.url_map.iter_rules()
        for method in rule.methods
    }

    assert ("/api/test-email", "POST") in rules
    assert ("/api/test-notification", "POST") in rules


def test_invalid_email_configuration_keeps_worker_unavailable_and_safe(
    vms_module,
    monkeypatch,
    capsys,
):
    names = (
        "MAIL_HOST",
        "MAIL_PORT",
        "MAIL_USERNAME",
        "MAIL_PASSWORD",
        "MAIL_FROM",
        "ALERT_EMAIL_TO",
        "MAIL_USE_TLS",
        "MAIL_USE_SSL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    assert vms_module.initialize_email_alerts() is False
    output = capsys.readouterr().out

    assert vms_module.email_alert_queue is None
    assert "[EMAIL][CONFIG] MAIL_HOST=missing" in output
    assert "MAIL_PASSWORD=missing" in output
    assert "result=configuration_error" in output


def test_email_test_route_requires_dashboard_authentication(vms_module):
    vms_module.app.secret_key = "unit-test-secret"
    client = vms_module.app.test_client()

    response = client.post("/api/test-email")

    assert response.status_code == 401
    assert response.get_json()["error"] == "Dashboard authentication required."
    assert response.get_json()["correlation_id"]


def test_email_test_route_returns_202_only_when_job_is_queued(
    vms_module,
    monkeypatch,
    tmp_path,
):
    vms_module.app.secret_key = "unit-test-secret"
    vms_module.email_alert_queue = object()
    monkeypatch.setattr(vms_module, "_default_camera", lambda: {
        "camera_id": "CAM001",
        "name": "Main Gate",
    })
    monkeypatch.setattr(vms_module, "_camera_snapshot", lambda _camera_id: {
        "frame": np.zeros((16, 16, 3), dtype=np.uint8),
    })
    monkeypatch.setattr(vms_module, "ALERT_IMAGE_DIR", tmp_path)
    monkeypatch.setattr(
        vms_module,
        "save_alert_image",
        lambda _frame: ("image-id", "http://localhost/alerts/image-id.jpg"),
    )
    queued_events = []
    monkeypatch.setattr(
        vms_module,
        "_queue_email_alert",
        lambda event, attachment_path=None: queued_events.append(event) or True,
    )
    client = vms_module.app.test_client()
    with client.session_transaction() as session:
        session["dashboard_user"] = {"username": "operator", "role": "user"}

    response = client.post("/api/test-email")
    payload = response.get_json()

    assert response.status_code == 202
    assert payload["success"] is True
    assert payload["correlation_id"] == queued_events[0]["correlation_id"]
    assert payload["event_id"] == "image-id"


def test_email_test_route_does_not_return_202_when_queue_rejects(
    vms_module,
    monkeypatch,
):
    vms_module.app.secret_key = "unit-test-secret"
    vms_module.email_alert_queue = object()
    monkeypatch.setattr(vms_module, "_default_camera", lambda: {
        "camera_id": "CAM001",
        "name": "Main Gate",
    })
    monkeypatch.setattr(vms_module, "_camera_snapshot", lambda _camera_id: {
        "frame": np.zeros((16, 16, 3), dtype=np.uint8),
    })
    monkeypatch.setattr(
        vms_module,
        "save_alert_image",
        lambda _frame: ("image-id", "http://localhost/alerts/image-id.jpg"),
    )
    monkeypatch.setattr(vms_module, "_queue_email_alert", lambda *_args, **_kwargs: False)
    client = vms_module.app.test_client()
    with client.session_transaction() as session:
        session["dashboard_user"] = {"username": "operator", "role": "user"}

    response = client.post("/api/test-email")

    assert response.status_code == 503
    assert response.get_json()["success"] is False
    assert response.get_json()["correlation_id"]


def test_email_test_route_sanitizes_queue_errors(
    vms_module,
    monkeypatch,
):
    vms_module.app.secret_key = "unit-test-secret"
    vms_module.email_alert_queue = object()
    monkeypatch.setattr(vms_module, "_default_camera", lambda: {
        "camera_id": "CAM001",
        "name": "Main Gate",
    })
    monkeypatch.setattr(vms_module, "_camera_snapshot", lambda _camera_id: {
        "frame": np.zeros((16, 16, 3), dtype=np.uint8),
    })
    monkeypatch.setattr(
        vms_module,
        "save_alert_image",
        lambda _frame: ("image-id", "http://localhost/alerts/image-id.jpg"),
    )
    monkeypatch.setattr(
        vms_module,
        "_queue_email_alert",
        Mock(side_effect=RuntimeError("SMTP credentials must not be returned")),
    )
    client = vms_module.app.test_client()
    with client.session_transaction() as session:
        session["dashboard_user"] = {"username": "operator", "role": "user"}

    response = client.post("/api/test-email")
    payload = response.get_json()

    assert response.status_code == 503
    assert payload["error"] == "Email job could not be queued."
    assert "SMTP credentials" not in response.get_data(as_text=True)


def test_unknown_email_persists_once_and_reuses_correlated_event(
    vms_module,
    monkeypatch,
):
    saved_events = []
    queued_events = []
    monkeypatch.setattr(
        vms_module.detection_database,
        "save",
        lambda event: saved_events.append(event.copy()) or 42,
    )
    monkeypatch.setattr(
        vms_module,
        "save_alert_image",
        lambda _frame: ("image-id", "http://localhost/alerts/image-id.jpg"),
    )
    monkeypatch.setattr(
        vms_module,
        "_queue_email_alert",
        lambda event, attachment_path=None: queued_events.append(event.copy()) or False,
    )
    track = {}
    face = type("Face", (), {"det_score": 0.9})()

    first_result = vms_module._send_face_only_unknown_email(
        "CAM001", "Main Gate", face, 0.1, np.zeros((4, 4, 3)), track=track
    )
    second_result = vms_module._send_face_only_unknown_email(
        "CAM001", "Main Gate", face, 0.1, np.zeros((4, 4, 3)), track=track
    )

    assert first_result is False
    assert second_result is False
    assert len(saved_events) == 1
    assert len(queued_events) == 2
    assert queued_events[0]["correlation_id"] == queued_events[1]["correlation_id"]
    assert queued_events[0]["id"] == queued_events[1]["id"] == 42


def test_unknown_email_queue_can_proceed_when_postgres_write_fails(
    vms_module,
    monkeypatch,
):
    monkeypatch.setattr(
        vms_module.detection_database,
        "save",
        Mock(side_effect=RuntimeError("database unavailable")),
    )
    monkeypatch.setattr(
        vms_module,
        "save_alert_image",
        lambda _frame: ("image-id", "http://localhost/alerts/image-id.jpg"),
    )
    queued_events = []
    monkeypatch.setattr(
        vms_module,
        "_queue_email_alert",
        lambda event, attachment_path=None: queued_events.append(event.copy()) or True,
    )
    face = type("Face", (), {"det_score": 0.9})()

    result = vms_module._send_face_only_unknown_email(
        "CAM001", "Main Gate", face, 0.1, np.zeros((4, 4, 3)), track={}
    )

    assert result is True
    assert len(queued_events) == 1
