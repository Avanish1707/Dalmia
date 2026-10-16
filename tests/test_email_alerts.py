import threading
from email import message_from_string

import pytest

from notifications.email import EmailAlertQueue, EmailConfig


def _set_mail_env(monkeypatch):
    values = {
        "MAIL_HOST": "smtp.example.test",
        "MAIL_PORT": "587",
        "MAIL_USERNAME": "alerts",
        "MAIL_PASSWORD": "test-only",
        "MAIL_FROM": "vms@example.test",
        "ALERT_EMAIL_TO": "first@example.test, second@example.test",
        "MAIL_USE_TLS": "true",
        "MAIL_USE_SSL": "false",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_email_config_is_optional_when_unconfigured(monkeypatch):
    for name in (
        "MAIL_HOST",
        "MAIL_PORT",
        "MAIL_USERNAME",
        "MAIL_PASSWORD",
        "MAIL_FROM",
        "ALERT_EMAIL_TO",
        "MAIL_USE_TLS",
        "MAIL_USE_SSL",
    ):
        monkeypatch.delenv(name, raising=False)

    assert EmailConfig.from_env() is None


def test_email_config_rejects_partial_configuration(monkeypatch):
    monkeypatch.setenv("MAIL_HOST", "smtp.example.test")

    with pytest.raises(ValueError, match="partially configured"):
        EmailConfig.from_env()


def test_email_config_parses_multiple_recipients_and_transport(monkeypatch):
    _set_mail_env(monkeypatch)

    config = EmailConfig.from_env()

    assert config.recipients == ("first@example.test", "second@example.test")
    assert config.use_tls is True
    assert config.use_ssl is False


def test_email_queue_sends_event_and_snapshot_off_worker_thread(tmp_path, monkeypatch):
    _set_mail_env(monkeypatch)
    config = EmailConfig.from_env()
    attachment = tmp_path / "evidence.jpg"
    attachment.write_bytes(b"jpeg evidence")
    sent = []
    thread_ids = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == (config.host, config.port, config.timeout)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def starttls(self):
            pass

        def login(self, username, password):
            assert (username, password) == (config.username, config.password)

        def sendmail(self, sender, recipients, message):
            thread_ids.append(threading.get_ident())
            sent.append((sender, recipients, message))

    queue = EmailAlertQueue(config, max_attempts=1, smtp_factory=FakeSMTP)
    camera_thread_id = threading.get_ident()
    queue.start()
    assert queue.submit(
        {
            "id": 27,
            "title": "Fall detected",
            "message": "Fall detected near entry",
            "detection_type": "fall_detected",
            "severity": "high",
            "camera_id": "CAM001",
            "camera_name": "Main Gate",
            "detected_at": "2026-10-16 11:00:00",
            "image_url": "https://example.test/alerts/evidence.jpg",
        },
        attachment_path=attachment,
    )
    queue._queue.join()
    queue.stop(timeout=1)

    assert len(sent) == 2
    assert [item[1] for item in sent] == [[recipient] for recipient in config.recipients]
    parsed = message_from_string(sent[0][2])
    body = parsed.get_payload(0).get_payload(decode=True).decode()
    attachment_part = parsed.get_payload(1)
    assert "Severity: high" in body
    assert "Camera ID: CAM001" in body
    assert "Evidence: https://example.test/alerts/evidence.jpg" in body
    assert attachment_part.get_filename() == "evidence.jpg"
    assert thread_ids[0] != camera_thread_id


def test_email_retry_only_retries_refused_recipients(monkeypatch):
    _set_mail_env(monkeypatch)
    config = EmailConfig.from_env()
    attempts = []
    message_ids = []

    class PartiallyRefusingSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def starttls(self):
            pass

        def login(self, *_):
            pass

        def sendmail(self, _sender, recipients, message):
            attempts.append(recipients[0])
            message_ids.append(message_from_string(message)["Message-ID"])
            if recipients[0] == config.recipients[1] and attempts.count(
                config.recipients[1]
            ) == 1:
                return {recipients[0]: (550, b"rejected")}
            return {}

    import notifications.email as email_module

    monkeypatch.setattr(email_module.time, "sleep", lambda _seconds: None)
    queue = EmailAlertQueue(config, max_attempts=3, smtp_factory=PartiallyRefusingSMTP)
    job = {
        "event": {"id": 99, "title": "Alert", "camera_id": "CAM001"},
        "attachment_path": None,
        "pending_recipients": list(config.recipients),
        "message_id": "stable-message-id",
    }

    queue._deliver_with_retries(job)

    assert attempts == [
        config.recipients[0],
        config.recipients[1],
        config.recipients[1],
    ]
    assert job["pending_recipients"] == []
    assert len(set(message_ids)) == 1


def test_email_delivery_failure_logs_stage_and_redacts_credentials(
    monkeypatch,
    capsys,
):
    _set_mail_env(monkeypatch)
    config = EmailConfig.from_env()

    def failing_smtp(*_args, **_kwargs):
        raise OSError(f"connection failed for {config.password}")

    queue = EmailAlertQueue(
        config,
        max_attempts=1,
        smtp_factory=failing_smtp,
    )
    queue._deliver_with_retries(
        {
            "event": {"id": "event-1"},
            "attachment_path": None,
            "pending_recipients": list(config.recipients),
            "message_id": "message-1",
        }
    )

    output = capsys.readouterr().out
    assert "[EMAIL][DELIVERY] stage=attempt" in output
    assert "[EMAIL][DELIVERY] stage=complete result=failed" in output
    assert "connection failed for [redacted]" in output
    assert config.password not in output
