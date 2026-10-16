"""Face-only temporal confirmation for unknown-person email alerts."""

from __future__ import annotations

import math

import numpy as np


class FaceOnlyUnknownTracker:
    def __init__(self, confirm_frames=2, track_timeout=5.0):
        self.confirm_frames = max(1, int(confirm_frames))
        self.track_timeout = max(0.1, float(track_timeout))
        self._tracks = []
        self._next_id = 1

    @staticmethod
    def _matches(current_box, previous_box):
        current = np.asarray(current_box, dtype=np.float32).reshape(4)
        previous = np.asarray(previous_box, dtype=np.float32).reshape(4)
        left = max(current[0], previous[0])
        top = max(current[1], previous[1])
        right = min(current[2], previous[2])
        bottom = min(current[3], previous[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        current_area = max(0.0, current[2] - current[0]) * max(
            0.0, current[3] - current[1]
        )
        previous_area = max(0.0, previous[2] - previous[0]) * max(
            0.0, previous[3] - previous[1]
        )
        union = current_area + previous_area - intersection
        if union > 0 and intersection / union >= 0.2:
            return True

        current_center = ((current[0] + current[2]) / 2, (current[1] + current[3]) / 2)
        previous_center = (
            (previous[0] + previous[2]) / 2,
            (previous[1] + previous[3]) / 2,
        )
        scale = max(current[2] - current[0], current[3] - current[1], 1.0)
        return math.dist(current_center, previous_center) <= scale * 0.75

    def update(self, box, identity_state, now):
        """Return (track, confirmed), retaining identity only across consecutive hits."""
        current_box = np.asarray(box, dtype=np.float32).reshape(4)
        self._tracks = [
            track
            for track in self._tracks
            if now - track["last_seen"] <= self.track_timeout
        ]
        for track in self._tracks:
            track["matched"] = False

        track = next(
            (
                candidate
                for candidate in self._tracks
                if not candidate["matched"]
                and self._matches(current_box, candidate["box"])
            ),
            None,
        )
        if track is None:
            track = {
                "track_id": self._next_id,
                "box": current_box.copy(),
                "identity_state": identity_state,
                "hits": 0,
                "last_seen": now,
                "email_attempted": False,
                "last_email_attempt_at": 0.0,
                "matched": True,
            }
            self._next_id += 1
            self._tracks.append(track)

        if identity_state == track["identity_state"]:
            track["hits"] += 1
        else:
            track["identity_state"] = identity_state
            track["hits"] = 1
            track["email_attempted"] = False
        track.update(box=current_box.copy(), last_seen=now, matched=True)
        return track, track["hits"] >= self.confirm_frames
