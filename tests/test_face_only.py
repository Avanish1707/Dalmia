import numpy as np
from types import SimpleNamespace

from face_only import FaceOnlyUnknownTracker, usable_face_box


def test_face_only_accepts_detector_box_without_landmark_geometry_gates():
    face = SimpleNamespace(
        bbox=np.array([100, 100, 150, 180]),
        kps=np.array([[120, 120], [135, 120], [127, 140]]),
    )

    box, reason = usable_face_box(face, (480, 640, 3))

    assert reason is None
    assert np.array_equal(box, face.bbox)


def test_face_only_clips_partial_box_to_frame():
    face = SimpleNamespace(bbox=np.array([-5, 10, 25, 50]))

    box, reason = usable_face_box(face, (100, 100, 3))

    assert reason is None
    assert np.array_equal(box, [0, 10, 25, 50])


def test_face_only_rejects_box_outside_frame_with_reason():
    face = SimpleNamespace(bbox=np.array([120, 10, 150, 50]))

    box, reason = usable_face_box(face, (100, 100, 3))

    assert box is None
    assert reason == "outside_frame"


def test_unknown_face_requires_two_matching_detections():
    tracker = FaceOnlyUnknownTracker(confirm_frames=2, track_timeout=5)
    box = np.array([10, 10, 50, 50])

    first, first_confirmed = tracker.update(box, "UNKNOWN", 1.0)
    second, second_confirmed = tracker.update(box + 1, "UNKNOWN", 1.5)

    assert first["track_id"] == second["track_id"]
    assert first_confirmed is False
    assert second_confirmed is True
    assert second["hits"] == 2


def test_face_tracker_resets_confirmation_when_identity_changes():
    tracker = FaceOnlyUnknownTracker(confirm_frames=2, track_timeout=5)
    box = np.array([10, 10, 50, 50])

    tracker.update(box, "UNKNOWN", 1.0)
    track, confirmed = tracker.update(box, "PENDING", 1.5)

    assert confirmed is False
    assert track["hits"] == 1
    assert track["email_attempted"] is False


def test_face_tracker_expires_a_departed_face():
    tracker = FaceOnlyUnknownTracker(confirm_frames=2, track_timeout=5)
    box = np.array([10, 10, 50, 50])

    first, _ = tracker.update(box, "UNKNOWN", 1.0)
    later, confirmed = tracker.update(box, "UNKNOWN", 7.0)

    assert later["track_id"] != first["track_id"]
    assert confirmed is False
