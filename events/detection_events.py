"""ROI entry/exit event tracking. Records only confirmed exits, never frames."""
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from face_association import is_validated_unknown_face


class DetectionEventManager:
    def __init__(self, database, image_dir, public_base_url, socketio,
                 email_sender=None,
                 dedup_seconds=10.0, exit_frame_offset=5,
                 camera_id=None, camera_name=None):
        self.database = database
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")
        self.socketio = socketio
        self.email_sender = email_sender
        self.exit_confirm_seconds = max(0.1, float(dedup_seconds))
        self.exit_frame_offset = max(0, int(exit_frame_offset))
        self.camera_id = camera_id
        self.camera_name = camera_name or camera_id or "Main Gate 01"
        self.active_person_events = []
        self.active_vehicle_events = []
        self._recent_event_cache = {}
        try:
            self.person_ids = json.loads(os.getenv("PERSON_IDS_JSON", "{}"))
        except json.JSONDecodeError:
            self.person_ids = {}

    def _camera_prefix(self):
        return f"[{self.camera_id}]" if self.camera_id else "[CAM]"

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _iou(a, b):
        left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _buffer_frame(frame):
        """
        Compress frame before storing it in the tracking buffer.
        """
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not success:
            raise RuntimeError("Could not encode tracking frame")

        return encoded.tobytes()

    @staticmethod
    def _decode_buffered_frame(frame_data):
        if isinstance(frame_data, bytes):
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                raise RuntimeError("Could not decode buffered frame")

            return frame

        # Backward compatibility
        return frame_data.copy()

    def _match_track(self, tracks, category, box):
        center, best, best_score = self._center(box), None, -1.0
        for track in tracks:
            if track["category"] != category or track["matched_this_frame"]:
                continue
            iou = self._iou(track["last_box"], box)
            distance = ((center[0] - track["center"][0]) ** 2 + (center[1] - track["center"][1]) ** 2) ** 0.5
            scale = max(1, box[2] - box[0], box[3] - box[1], track["last_box"][2] - track["last_box"][0], track["last_box"][3] - track["last_box"][1])
            if iou < .10 and distance > max(100, scale * 1.5):
                continue
            score = iou - distance / (scale * 10)
            if score > best_score:
                best, best_score = track, score
        return best

    def _update_track(self, tracks, category, event, frame, now):
        is_person = category == "person"
        incoming_state = None
        identity_name = event.get("person_name", "Unknown")
        has_face = bool(event.get("annotation_box"))
        if is_person:
            incoming_state = event.get("recognition_state")
            if incoming_state not in {"PENDING", "UNKNOWN", "KNOWN"}:
                if event.get("detection_type") == "known_person":
                    incoming_state = "KNOWN"
                elif identity_name == "Pending":
                    incoming_state = "PENDING"
                elif (
                    event.get("detection_type") == "unknown_person"
                    and is_validated_unknown_face(
                        event.get("face_validation"),
                        event.get("recognition_decision"),
                    )
                ):
                    incoming_state = "UNKNOWN"
                else:
                    incoming_state = "PENDING"
            unknown_validated = (
                incoming_state == "UNKNOWN"
                and is_validated_unknown_face(
                    event.get("face_validation"),
                    event.get("recognition_decision"),
                )
            )
            if incoming_state == "UNKNOWN" and not unknown_validated:
                incoming_state = "PENDING"
            event["recognition_state"] = incoming_state
            event["detection_type"] = (
                "known_person" if incoming_state == "KNOWN" else "unknown_person"
            )
        track_id = event.get("track_id")
        track = next(
            (
                item for item in tracks
                if track_id is not None
                and item["category"] == category
                and item.get("track_id") == track_id
                and not item["matched_this_frame"]
            ),
            None,
        )
        if track is None and track_id is None:
            track = self._match_track(tracks, category, event["box"])
        if track is None:
            track = {
                "event_id": str(uuid.uuid4()),
                "session_id": str(uuid.uuid4()),
                "track_id": event.get("track_id"),
                "category": category,
                "state": "ENTERED_ROI",
                "recognition_state": incoming_state if is_person else None,
                "unknown_validated": bool(is_person and unknown_validated),
                "known_person_id": event.get("person_id") if incoming_state == "KNOWN" else None,
                "known_person_name": event.get("person_name") if incoming_state == "KNOWN" else None,
                "known_confidence": event.get("confidence") if incoming_state == "KNOWN" else None,
                "alert_generated": False,
                "event_finalized": False,
                "inside_roi": True,
                "first_seen": now,
                "last_seen": now,
                "last_box": event["box"],
                "center": self._center(event["box"]),
                "matched_this_frame": True,
                "known_detected_once": is_person and incoming_state == "KNOWN",
                "no_face_logged": False,
                "vehicle_context_detected": bool(event.get("vehicle_context_detected")),
                "suppress_notify": bool(event.get("suppress_notify")),
                "event": event.copy(),
                "frames": deque(maxlen=self.exit_frame_offset + 2),
            }
            track["event"].update({
                "event_id": track["event_id"],
                "session_id": track["session_id"],
                "track_id": track["track_id"],
                "alert_generated": False,
                "event_finalized": False,
                "inside_roi": True,
            })
            tracks.append(track)
            print(
                f"{self._camera_prefix()} [FACE EVENT] "
                f"track={track['track_id'] or track['event_id']} ENTER_ROI "
                f"state={incoming_state}"
            )
            if is_person and incoming_state == "KNOWN":
                print(
                    f"{self._camera_prefix()} [FACE EVENT] "
                    f"track={track['track_id'] or track['event_id']} "
                    f"recognition=KNOWN name={track['known_person_name']}"
                )
        else:
            previous_state = track.get("recognition_state", "PENDING")
            recognition_state = previous_state
            if is_person:
                if previous_state == "KNOWN" or incoming_state == "KNOWN":
                    recognition_state = "KNOWN"
                elif (
                    previous_state == "UNKNOWN"
                    and track.get("unknown_validated")
                ) or unknown_validated:
                    recognition_state = "UNKNOWN"
                else:
                    recognition_state = "PENDING"
            track.update({
                "state": "TRACKING",
                "last_seen": now,
                "last_box": event["box"],
                "center": self._center(event["box"]),
                "matched_this_frame": True,
                "track_id": event.get("track_id") or track.get("track_id"),
                "inside_roi": True,
            })
            if is_person:
                track["recognition_state"] = recognition_state
                track["unknown_validated"] = bool(
                    track.get("unknown_validated") or unknown_validated
                )
                track["known_detected_once"] = recognition_state == "KNOWN"
                if recognition_state != previous_state:
                    print(
                        f"{self._camera_prefix()} [FACE EVENT] "
                        f"track={track['track_id'] or track['event_id']} "
                        f"recognition={recognition_state}"
                    )
                    if recognition_state == "KNOWN":
                        print(
                            f"{self._camera_prefix()} [FACE EVENT] "
                            f"track={track['track_id'] or track['event_id']} "
                            f"recognition=KNOWN name={event.get('person_name')}"
                        )
        if is_person:
            if incoming_state == "KNOWN":
                if event.get("person_id") is not None:
                    track["known_person_id"] = event["person_id"]
                if event.get("person_name") not in {None, "", "Unknown", "Pending"}:
                    track["known_person_name"] = event["person_name"]
                if event.get("confidence") is not None:
                    track["known_confidence"] = event["confidence"]
            event["recognition_state"] = track["recognition_state"]
            event["detection_type"] = (
                "known_person"
                if track["recognition_state"] == "KNOWN"
                else "unknown_person"
            )
            if track["recognition_state"] == "KNOWN":
                event.update({
                    "person_id": track["known_person_id"],
                    "person_name": track["known_person_name"],
                    "confidence": track["known_confidence"],
                    "title": "Known Person Detected",
                    "message": f"{track['known_person_name']} detected at {event['gate_name']}",
                })
            else:
                event.update({"person_id": None, "person_name": "Pending"})
            if has_face:
                track["no_face_logged"] = False
            elif not track.get("no_face_logged"):
                print(
                    f"{self._camera_prefix()} [FACE EVENT] "
                    f"track={track['track_id'] or track['event_id']} "
                    "recognition=PENDING_NO_FACE"
                )
                track["no_face_logged"] = True
        event["event_id"] = track["event_id"]
        event["session_id"] = track["session_id"]
        event["track_id"] = track.get("track_id")
        event["alert_generated"] = track["alert_generated"]
        event["event_finalized"] = track["event_finalized"]
        event["inside_roi"] = True
        event["unknown_validated"] = track.get("unknown_validated", False)
        track["event"] = event.copy()
        track["vehicle_context_detected"] = track.get("vehicle_context_detected", False) or bool(event.get("vehicle_context_detected"))
        track["suppress_notify"] = track.get("suppress_notify", False) or bool(event.get("suppress_notify"))
        track["event"]["box"] = event["box"]
        if track.get("vehicle_context_detected"):
            track["event"]["vehicle_context_detected"] = True
        if track.get("suppress_notify"):
            track["event"]["suppress_notify"] = True
        try:
            buffered_frame = self._buffer_frame(frame)
        except Exception as error:
            print(f"{self._camera_prefix()} [ERROR] Could not buffer tracking frame: {error}")
            return
        track["frames"].append({"frame": buffered_frame, "event": track["event"].copy()})

    def _save_image(self, frame):
        filename = f"{uuid.uuid4()}.jpg"
        if not cv2.imwrite(str(self.image_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise RuntimeError("Could not save detection image")
        return f"{self.public_base_url}/alerts/{filename}"

    @staticmethod
    def _draw_event(frame, event):
        if event["detection_type"] == "fall_detected":
            x1, y1, x2, y2 = event["box"]
            label, color = "FALL DETECTED", (0, 0, 255)
        elif event["detection_type"] == "helmet_detected":
            x1, y1, x2, y2 = event["box"]
            label, color = "HELMET DETECTED", (0, 215, 255)
        elif event["detection_type"] == "object_theft":
            box = event.get("annotation_box") or event.get("box")
            if not box:
                return
            label, color = f"DRUM_CONTAINER {event.get('confidence', 0.0):.2f}", (0, 255, 255)
        elif event["detection_type"] == "vehicle":
            x1, y1, x2, y2 = event["box"]
            label, color = f"{event['vehicle_type'].replace('_', ' ').title()} | {event['vehicle_class']} | {event['confidence']:.2f}", (0, 140, 255)
        elif event["detection_type"] == "known_person":
            box = event.get("annotation_box") or event.get("box")
            if not box:
                return
            x1, y1, x2, y2 = box
            label, color = f"KNOWN | {event['person_name']} | ID: {event.get('person_id') or 'N/A'} | {event['confidence']:.2f}", (0, 255, 0)
        else:
            box = event.get("annotation_box") or event.get("box")
            if not box:
                return
            x1, y1, x2, y2 = box
            count = event.get("unknown_count", 1)
            label, color = ("UNKNOWN" if count == 1 else f"UNKNOWN PERSONS: {count}"), (0, 0, 255)
        if event["detection_type"] == "object_theft":
            x1, y1, x2, y2 = event["annotation_box"] or event["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_color = (
            (0, 0, 0)
            if event["detection_type"] == "known_person"
            else (255, 255, 255)
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            .65,
            3,
        )
        label_bottom = max(text_height + baseline + 10, y1)
        label_top = max(0, label_bottom - text_height - baseline - 10)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 12, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_bottom - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            .65,
            text_color,
            3,
            cv2.LINE_AA,
        )

    def _dedup_key(self, event):
        detection_type = event.get("detection_type")
        camera_id = event.get("camera_id") or self.camera_id
        gate_name = event.get("gate_name")
        if detection_type == "known_person":
            return (
                "known_person",
                camera_id,
                event.get("event_id") or event.get("session_id") or event.get("track_id"),
            )
        if detection_type == "vehicle":
            return (
                "vehicle",
                camera_id,
                gate_name,
                event.get("vehicle_type"),
                event.get("vehicle_class"),
            )
        if detection_type == "fall_detected":
            return ("fall_detected", camera_id, event.get("track_id"))
        if detection_type == "helmet_detected":
            return ("helmet_detected", camera_id, event.get("track_id"))
        if detection_type == "object_theft":
            return (
                "object_theft",
                camera_id,
                event.get("track_id"),
                event.get("session_id"),
            )
        if detection_type == "unknown_person" and (
            event.get("event_id") or event.get("session_id") or event.get("track_id")
        ):
            return (
                "unknown_person",
                camera_id,
                event.get("event_id") or event.get("session_id") or event.get("track_id"),
            )
        return (
            detection_type,
            camera_id,
            gate_name,
        )

    def _is_duplicate_event(self, event):
        now = time.time()
        key = self._dedup_key(event)
        previous = self._recent_event_cache.get(key)
        if previous is not None and (now - previous) < self.exit_confirm_seconds:
            if event.get("detection_type") == "object_theft":
                print(
                    "[OBJECT-THEFT] Duplicate event suppressed "
                    f"camera={event.get('camera_id', self.camera_id)} "
                    f"track={event.get('track_id')} "
                    f"session={event.get('session_id')}"
                )
            return True
        self._recent_event_cache[key] = now
        return False

    def publish_immediate(self, event, frame, notify=True):
        """Publish a confirmed non-exit event through existing persistence/IO."""
        image = frame.copy()
        self._draw_event(image, event)
        return self._publish(event, image, notify=notify)

    def _notify(self, event):
        if self.email_sender is not None:
            filename = Path(urlsplit(event.get("image_url", "")).path).name
            attachment_path = self.image_dir / filename if filename else None
            response = self.email_sender(
                event=event,
                attachment_path=attachment_path,
            )
            if event["detection_type"] == "unknown_person":
                queued = bool(response)
                print(
                    f"{self._camera_prefix()} [UNKNOWN] EMAIL_QUEUE_RESULT "
                    f"| queued={queued}"
                )
            return response
        print("[WARNING] No email sender configured; alert email not queued.")

    def _publish(self, event, image, notify):
        if event.get("detection_type") == "unknown_person" and not (
            event.get("recognition_state") == "UNKNOWN"
            and is_validated_unknown_face(
                event.get("face_validation"),
                event.get("recognition_decision"),
            )
        ):
            print(
                f"{self._camera_prefix()} [FACE EVENT] "
                "UNKNOWN_REJECTED reason=missing_face_validation"
            )
            return None
        if self._is_duplicate_event(event):
            return None

        event["image_url"] = self._save_image(image)
        event["camera_id"] = event.get("camera_id") or self.camera_id
        event["camera_name"] = event.get("camera_name") or self.camera_name
        event["id"] = self.database.save(event)
        event["type"], event["gate"] = event["detection_type"], event["gate_name"]
        event["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["detected_at"]))
        if event["detection_type"] != "known_person":
            self.socketio.emit("detection_event", event)
        if event["detection_type"] == "unknown_person":
            self.socketio.emit("face_alert", event)
            print(
                f"{self._camera_prefix()} [UNKNOWN] ALERT_SENT "
                "| channel=socketio"
            )
        if notify and event["detection_type"] != "known_person":
            try:
                self._notify(event)
            except Exception as error:
                print(f"{self._camera_prefix()} [ERROR] Detection notification failed: {error}")
        return event

    def _finalize_expired(self, tracks, now):
        expired, remaining = [], []
        for track in tracks:
            if now - track["last_seen"] < self.exit_confirm_seconds:
                remaining.append(track)
            elif track["frames"]:
                track["state"] = "FINALIZED"
                # With frames 101..108 and a confirmed exit at 108, index -7
                # is frame 102: exactly EXIT_FRAME - 6.
                index = (
                    max(-len(track["frames"]), -(self.exit_frame_offset + 1))
                    if self.exit_frame_offset
                    else -1
                )
                expired.append((track, track["frames"][index]))
        tracks[:] = remaining
        return expired

    def _person_event(self, face, gate_name, now):
        name = getattr(face, "recognized_name", "Unknown")
        recognition_state = getattr(face, "recognition_state", None)
        validation = getattr(face, "face_validation", None)
        recognition_decision = getattr(face, "recognition_decision", None)
        if recognition_state not in {"PENDING", "UNKNOWN", "KNOWN"}:
            recognition_state = (
                "KNOWN"
                if name not in {"Unknown", "Pending"}
                else "UNKNOWN"
                if is_validated_unknown_face(validation, recognition_decision)
                else "PENDING"
            )
        if recognition_state == "UNKNOWN" and not (
            is_validated_unknown_face(validation, recognition_decision)
        ):
            recognition_state = "PENDING"
        if recognition_state == "PENDING":
            name = "Pending"
        elif recognition_state == "UNKNOWN":
            name = "Unknown"
        confidence = float(getattr(face, "recognition_score", 0.0))
        body_box = getattr(face, "associated_person_box", None)
        if body_box is None:
            body_box = getattr(face, "bbox", None)
        annotation_box = getattr(face, "annotation_box", None)
        if annotation_box is None and getattr(face, "associated_person_box", None) is None:
            annotation_box = getattr(face, "bbox", None)
        known = recognition_state == "KNOWN"
        return {"detected_at": now, "recognition_state": recognition_state, "recognition_decision": recognition_decision, "face_validation": validation, "detection_type": "known_person" if known else "unknown_person", "person_id": (getattr(face, "person_id", None) or self.person_ids.get(name)) if known else None, "person_name": name, "confidence": confidence, "gate_name": gate_name, "camera_id": self.camera_id, "camera_name": self.camera_name, "track_id": getattr(face, "track_id", None), "box": tuple(int(v) for v in body_box) if body_box is not None else None, "annotation_box": tuple(int(v) for v in annotation_box) if annotation_box is not None else None, "body_box": tuple(int(v) for v in body_box) if body_box is not None else None, "title": "Known Person Detected" if known else "Unknown Person Detected", "message": f"{name} detected at {gate_name}"}

    def _publish_unknowns(self, unknowns, gate_name, now):
        eligible_unknowns = [
            (track, chosen)
            for track, chosen in unknowns
            if (
                track.get("recognition_state") == "UNKNOWN"
                and track.get("unknown_validated")
                and not track.get("alert_generated")
                and not track.get("inside_roi", False)
                and is_validated_unknown_face(
                    track.get("event", {}).get("face_validation"),
                    track.get("event", {}).get("recognition_decision"),
                )
            )
        ]
        if not eligible_unknowns:
            return None
        track, chosen = eligible_unknowns[0]
        event = track["event"].copy()
        count = len(eligible_unknowns)
        vehicle_context = any(
            item[0].get("vehicle_context_detected")
            for item in eligible_unknowns
        )
        event.update({"detected_at": now, "detection_type": "unknown_person", "recognition_state": "UNKNOWN", "person_name": "Unknown", "person_id": None, "camera_id": self.camera_id, "camera_name": self.camera_name, "unknown_count": count, "title": "Unknown Person Detected" if count == 1 else "Unknown Persons Detected", "message": f"Unknown person detected at {gate_name}" if count == 1 else f"{count} unknown persons detected at {gate_name}", "alert_generated": True, "event_finalized": True, "inside_roi": False})
        if vehicle_context:
            event["title"] = "Unknown Person Detected at Vehicle" if count == 1 else "Unknown Persons Detected at Vehicle"
            event["message"] = f"Unknown person detected at vehicle at {gate_name}" if count == 1 else f"{count} unknown persons detected at vehicle at {gate_name}"
            event["vehicle_context_detected"] = True
        try:
            image = self._decode_buffered_frame(chosen["frame"])
        except Exception as error:
            print(f"{self._camera_prefix()} [ERROR] Could not decode tracking frame: {error}")
            return None
        for item, _ in eligible_unknowns:
            marked = item["event"].copy()
            marked["unknown_count"] = count
            self._draw_event(image, marked)
        print(
            f"{self._camera_prefix()} [UNKNOWN] ALERT_CONFIRMED "
            f"| people={count}"
        )
        for item, _ in eligible_unknowns:
            item["alert_generated"] = True
            item["event"]["alert_generated"] = True
        published = self._publish(event, image, notify=True)
        if published is None:
            print(
                f"{self._camera_prefix()} [FACE EVENT] "
                f"track={track['track_id'] or track['event_id']} "
                "DUPLICATE_ALERT_SUPPRESSED"
            )
        else:
            print(
                f"{self._camera_prefix()} [FACE EVENT] "
                f"track={track['track_id'] or track['event_id']} "
                "UNKNOWN_ALERT_PUBLISHED"
            )
        return published

    def process_frame(self, frame, faces, vehicles, gate_name, detected_at, alert_frame=None):
        """Ingest one annotated ROI frame; return records finalized on this call."""
        annotated = alert_frame if alert_frame is not None else frame
        for track in self.active_person_events + self.active_vehicle_events:
            track["matched_this_frame"] = False
        for face in faces:
            self._update_track(self.active_person_events, "person", self._person_event(face, gate_name, detected_at), annotated, detected_at)
        for vehicle in vehicles:
            vehicle_type = vehicle["vehicle_type"]
            title = "Two Wheeler Detected" if vehicle_type == "two_wheeler" else "Four Wheeler Detected"
            event = {"detected_at": detected_at, "detection_type": "vehicle", "vehicle_type": vehicle_type, "vehicle_class": vehicle["class_name"], "confidence": float(vehicle["confidence"]), "gate_name": gate_name, "camera_id": self.camera_id, "camera_name": self.camera_name, "box": tuple(int(v) for v in vehicle["box"]), "title": title, "message": f"{title} at {gate_name}"}
            self._update_track(self.active_vehicle_events, f"vehicle:{vehicle_type}", event, annotated, detected_at)
        active_unknown_tracks = [
            track
            for track in self.active_person_events
            if not track["known_detected_once"]
        ]
        if active_unknown_tracks and self.active_vehicle_events:
            for track in active_unknown_tracks:
                track["vehicle_context_detected"] = True
                track["event"]["vehicle_context_detected"] = True
            for track in self.active_vehicle_events:
                track["suppress_notify"] = True
                track["event"]["suppress_notify"] = True
        # Keep the buffer aligned with real processing frames while a track is
        # temporarily missing. This makes the selection relative to the
        # confirmation frame (rather than merely the fifth prior detection).
        for track in self.active_person_events + self.active_vehicle_events:
            if not track["matched_this_frame"]:
                try:
                    buffered_frame = self._buffer_frame(annotated)
                except Exception as error:
                    print(f"[ERROR] Could not buffer tracking frame: {error}")
                    continue
                track["frames"].append({
                    "frame": buffered_frame,
                    "event": track["event"].copy(),
                })
        people = self._finalize_expired(self.active_person_events, detected_at)
        vehicles = self._finalize_expired(self.active_vehicle_events, detected_at)
        records = []
        unknowns = []
        for track, chosen in people:
            state = track.get("recognition_state")
            track_label = track["track_id"] or track["event_id"]
            track["inside_roi"] = False
            print(
                f"{self._camera_prefix()} [FACE EVENT] "
                f"track={track_label} EXIT_ROI"
            )
            track["event_finalized"] = True
            track["event"].update({
                "recognition_state": state,
                "inside_roi": False,
                "event_finalized": True,
            })
            event = track["event"].copy()
            event["detected_at"] = detected_at
            if state == "KNOWN":
                event.update({
                    "detection_type": "known_person",
                    "person_id": track["known_person_id"],
                    "person_name": track["known_person_name"],
                    "confidence": track["known_confidence"],
                })
                try:
                    image = self._decode_buffered_frame(chosen["frame"])
                except Exception as error:
                    print(f"{self._camera_prefix()} [ERROR] Could not decode tracking frame: {error}")
                    continue
                self._draw_event(image, event)
                records.append(self._publish(event, image, notify=False))
            elif state == "UNKNOWN" and track.get("unknown_validated") and not track["alert_generated"]:
                track["event"]["detection_type"] = "unknown_person"
                unknowns.append((track, chosen))
        if unknowns:
            records.append(self._publish_unknowns(unknowns, gate_name, detected_at))
        for track, chosen in vehicles:
            event = track["event"].copy(); event["detected_at"] = detected_at
            try:
                image = self._decode_buffered_frame(chosen["frame"])
            except Exception as error:
                print(f"{self._camera_prefix()} [ERROR] Could not decode tracking frame: {error}")
                continue
            self._draw_event(image, event)
            records.append(self._publish(event, image, notify=not track.get("suppress_notify", False)))
        return records
