"""Bounded asynchronous email delivery for finalized detection alerts."""

import os
import queue
import smtplib
import threading
import time
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable


def _parse_bool(name, value):
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipients: tuple[str, ...]
    use_tls: bool
    use_ssl: bool
    timeout: float = 10.0

    @classmethod
    def from_env(cls):
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
        configured = {name: os.getenv(name, "").strip() for name in names}
        core_fields = names[:6]
        use_tls = _parse_bool("MAIL_USE_TLS", configured["MAIL_USE_TLS"])
        use_ssl = _parse_bool("MAIL_USE_SSL", configured["MAIL_USE_SSL"])
        if use_tls and use_ssl:
            raise ValueError("MAIL_USE_TLS and MAIL_USE_SSL cannot both be true")
        if not any(configured[name] for name in core_fields):
            if use_tls or use_ssl:
                raise ValueError(
                    "Email transport security is configured but mail settings are missing"
                )
            return None

        required = ("MAIL_HOST", "MAIL_PORT", "MAIL_FROM", "ALERT_EMAIL_TO")
        missing = [name for name in required if not configured[name]]
        if missing:
            raise ValueError(
                "Email alerts are partially configured; set "
                + ", ".join(missing)
            )

        username, password = (
            configured["MAIL_USERNAME"],
            configured["MAIL_PASSWORD"],
        )
        if bool(username) != bool(password):
            raise ValueError(
                "MAIL_USERNAME and MAIL_PASSWORD must be set together"
            )

        try:
            port = int(configured["MAIL_PORT"])
        except ValueError as exc:
            raise ValueError("MAIL_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("MAIL_PORT must be between 1 and 65535")

        sender = configured["MAIL_FROM"]
        recipients = tuple(
            recipient.strip()
            for recipient in configured["ALERT_EMAIL_TO"].split(",")
            if recipient.strip()
        )
        if not recipients:
            raise ValueError("ALERT_EMAIL_TO must contain at least one recipient")
        addresses = (sender, *recipients)
        if any(
            address.count("@") != 1
            or any(character.isspace() for character in address)
            or address.startswith("@")
            or address.endswith("@")
            for address in addresses
        ):
            raise ValueError("MAIL_FROM and ALERT_EMAIL_TO must contain valid email addresses")

        return cls(
            host=configured["MAIL_HOST"],
            port=port,
            username=username,
            password=password,
            sender=sender,
            recipients=recipients,
            use_tls=use_tls,
            use_ssl=use_ssl,
        )


class EmailAlertQueue:
    """Send alert messages off the camera thread with bounded work and retries."""

    def __init__(
        self,
        config,
        max_queue_size=256,
        max_attempts=3,
        smtp_factory: Callable | None = None,
    ):
        self.config = config
        self.max_attempts = max(1, int(max_attempts))
        self._smtp_factory = smtp_factory
        self._queue = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self._thread = None
        self._stopping = threading.Event()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="email-alert-worker",
            daemon=True,
        )
        self._thread.start()
        print("[EMAIL][WORKER] stage=start result=ready")

    def submit(self, event, attachment_path=None):
        if self._thread is None or not self._thread.is_alive() or self._stopping.is_set():
            print(
                "[EMAIL][QUEUE] result=failed "
                f"event_id={event.get('id', 'unknown')} "
                "reason=worker_not_running"
            )
            return False
        job = {
            "event": dict(event),
            "attachment_path": Path(attachment_path) if attachment_path else None,
            "pending_recipients": list(self.config.recipients),
        }
        event_id = str(event.get("id") or uuid.uuid4())
        job["message_id"] = hashlib.sha256(
            event_id.encode("utf-8")
        ).hexdigest()
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            print(
                "[EMAIL][QUEUE] result=failed "
                f"event_id={event.get('id', 'unknown')} reason=queue_full"
            )
            return False
        print(
            "[EMAIL][QUEUE] result=accepted "
            f"event_id={event.get('id', 'unknown')} "
            f"recipients={len(job['pending_recipients'])}"
        )
        return True

    def stop(self, timeout=20):
        if self._thread is None:
            return
        self._stopping.set()
        self._thread.join(timeout=max(0.0, float(timeout)))
        if self._thread.is_alive():
            print("[EMAIL][WORKER] stage=stop result=timeout")
        else:
            print("[EMAIL][WORKER] stage=stop result=complete")

    def _run(self):
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue

            try:
                self._deliver_with_retries(job)
            finally:
                self._queue.task_done()

    def _safe_error_detail(self, error):
        detail = str(error)
        for sensitive_value in (
            self.config.username,
            self.config.password,
        ):
            if sensitive_value:
                detail = detail.replace(sensitive_value, "[redacted]")
        return detail.replace("\r", " ").replace("\n", " ")[:240]

    def _deliver_with_retries(self, job):
        event = job["event"]
        event_id = event.get("id", "unknown")
        for attempt in range(1, self.max_attempts + 1):
            try:
                transport = (
                    "ssl"
                    if self.config.use_ssl
                    else "starttls"
                    if self.config.use_tls
                    else "plain"
                )
                print(
                    "[EMAIL][DELIVERY] stage=attempt "
                    f"event_id={event_id} attempt={attempt}/{self.max_attempts} "
                    f"host={self.config.host} port={self.config.port} "
                    f"transport={transport} "
                    f"recipients={len(job['pending_recipients'])}"
                )
                self._send(job)
                print(
                    f"[EMAIL][DELIVERY] stage=complete result=sent "
                    f"event_id={event_id}"
                )
                return
            except Exception as error:
                print(
                    "[EMAIL][DELIVERY] stage=complete result=failed "
                    f"event_id={event_id} attempt={attempt}/{self.max_attempts} "
                    f"error_type={type(error).__name__} "
                    f"detail={self._safe_error_detail(error)}"
                )
                if attempt < self.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))

    def _send(self, job):
        event = job["event"]
        title = str(event.get("title") or event.get("detection_type") or "VMS alert")
        camera = str(event.get("camera_name") or event.get("gate_name") or "Unknown camera")
        subject = f"{title} - {camera}"
        message = MIMEMultipart()
        message["Subject"] = subject.replace("\r", " ").replace("\n", " ")
        message["From"] = self.config.sender
        message["To"] = ", ".join(self.config.recipients)
        message["Message-ID"] = f"<vms-alert-{job['message_id']}@dalmia-vms>"
        message.attach(MIMEText(self._format_body(event), "plain", "utf-8"))

        attachment_path = job["attachment_path"]
        if attachment_path is not None and attachment_path.is_file():
            with attachment_path.open("rb") as evidence:
                attachment = MIMEApplication(evidence.read(), _subtype="jpeg")
            attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=attachment_path.name,
            )
            message.attach(attachment)
        elif attachment_path is not None:
            print(
                "[WARNING] Email alert evidence file unavailable; "
                f"event_id={event.get('id', 'unknown')}"
            )

        if self._smtp_factory is not None:
            smtp_client = self._smtp_factory
        elif self.config.use_ssl:
            smtp_client = smtplib.SMTP_SSL
        else:
            smtp_client = smtplib.SMTP

        with smtp_client(
            self.config.host,
            self.config.port,
            timeout=self.config.timeout,
        ) as smtp:
            if self.config.use_tls:
                smtp.starttls()
            if self.config.username:
                smtp.login(self.config.username, self.config.password)
            failed_recipients = []
            for recipient in job["pending_recipients"]:
                try:
                    refused = smtp.sendmail(
                        self.config.sender,
                        [recipient],
                        message.as_string(),
                    )
                except (OSError, smtplib.SMTPException) as error:
                    print(
                        "[EMAIL][RECIPIENT] result=failed "
                        f"event_id={event.get('id', 'unknown')} "
                        f"error_type={type(error).__name__} "
                        f"detail={self._safe_error_detail(error)}"
                    )
                    failed_recipients.append(recipient)
                    continue
                if refused:
                    status = refused.get(recipient, (None, b""))[0]
                    response = refused.get(recipient, (None, b""))[1]
                    failed_recipients.append(recipient)
                    print(
                        "[EMAIL][RECIPIENT] result=refused "
                        f"event_id={event.get('id', 'unknown')} "
                        f"smtp_status={status} "
                        f"detail={self._safe_error_detail(Exception(response))}"
                    )

        job["pending_recipients"] = failed_recipients
        if failed_recipients:
            raise smtplib.SMTPException(
                f"Delivery failed for {len(failed_recipients)} recipient(s)"
            )

    @staticmethod
    def _format_body(event):
        lines = [
            str(event.get("title") or event.get("detection_type") or "VMS alert"),
            str(event.get("message") or ""),
        ]
        severity = event.get("severity")
        if severity is not None:
            lines.append(f"Severity: {severity}")

        camera_id = event.get("camera_id")
        camera_name = event.get("camera_name") or event.get("gate_name")
        detected_at = event.get("detected_at")
        if camera_id is not None:
            lines.append(f"Camera ID: {camera_id}")
        if camera_name is not None:
            lines.append(f"Camera name: {camera_name}")
        if detected_at is not None:
            if isinstance(detected_at, (int, float)):
                try:
                    detected_at = datetime.fromtimestamp(detected_at).astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    )
                except (OSError, OverflowError, ValueError):
                    detected_at = str(detected_at)
            lines.append(f"Event timestamp: {detected_at}")

        for label, key in (
            ("Detection type", "detection_type"),
            ("Person", "person_name"),
            ("Person ID", "person_id"),
            ("Vehicle type", "vehicle_type"),
            ("Vehicle class", "vehicle_class"),
            ("Confidence", "confidence"),
            ("Unknown count", "unknown_count"),
            ("Track ID", "track_id"),
            ("ROI", "roi"),
            ("ROI name", "roi_name"),
            ("Inside ROI", "inside_roi"),
            ("Exit", "exit_info"),
        ):
            value = event.get(key)
            if value is not None and value != "":
                lines.append(f"{label}: {value}")

        image_url = event.get("image_url")
        if image_url:
            lines.append(f"Evidence: {image_url}")
        return "\n".join(line for line in lines if line)
