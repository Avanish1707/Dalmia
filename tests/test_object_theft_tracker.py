from object_theft.detector import ObjectTheftDetector
from object_theft.tracker import ObjectTheftTracker


def test_object_theft_detector_is_pinned_to_cpu(monkeypatch):
    monkeypatch.setattr("object_theft.detector.OBJECT_THEFT_ENABLED", False)

    detector = ObjectTheftDetector()

    assert detector.device == "cpu"


def _track(box, confidence=0.9):
    return {
        "class_name": "DRUM_CONTAINER",
        "canonical_class": "DRUM_CONTAINER",
        "bbox": box,
        "confidence": confidence,
    }


def test_visible_object_outside_roi_does_not_count_as_theft():
    tracker = ObjectTheftTracker(confirm_frames=2, max_missed=2)
    roi = [(0, 0), (300, 0), (300, 300), (0, 300)]
    box = (50, 50, 120, 120)

    for _ in range(2):
        assert tracker.update("CAM008", [_track(box)], roi_polygon=roi, now=1.0) == []

    outside = (350, 70, 430, 150)
    assert tracker.update("CAM008", [_track(outside)], roi_polygon=roi, now=2.0) == []
    assert tracker.update("CAM008", [_track(outside)], roi_polygon=roi, now=3.0) == []


def test_disappearance_after_confirmed_inside_roi_requires_missing_confirmation():
    tracker = ObjectTheftTracker(confirm_frames=2, max_missed=3)
    roi = [(0, 0), (300, 0), (300, 300), (0, 300)]
    box = (50, 50, 120, 120)

    for _ in range(2):
        assert tracker.update("CAM008", [_track(box)], roi_polygon=roi, now=1.0) == []

    assert tracker.update("CAM008", [], roi_polygon=roi, now=2.0) == []
    assert tracker.update("CAM008", [], roi_polygon=roi, now=3.0) == []

    # The object is only confirmed as removed once it has been missing for the full threshold.
    assert len(tracker.update("CAM008", [], roi_polygon=roi, now=4.0)) == 1


def test_visible_outside_roi_after_missing_is_not_theft():
    tracker = ObjectTheftTracker(confirm_frames=2, max_missed=2)
    roi = [(0, 0), (300, 0), (300, 300), (0, 300)]
    box = (50, 50, 120, 120)
    outside = (350, 70, 430, 150)

    for _ in range(2):
        assert tracker.update("CAM008", [_track(box)], roi_polygon=roi, now=1.0) == []

    assert tracker.update("CAM008", [], roi_polygon=roi, now=2.0) == []
    assert tracker.update("CAM008", [_track(outside)], roi_polygon=roi, now=3.0) == []
    assert tracker.update("CAM008", [], roi_polygon=roi, now=4.0) == []
    assert tracker.update("CAM008", [], roi_polygon=roi, now=5.0) == []
