import numpy as np

from face_only import FaceOnlyUnknownTracker


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
