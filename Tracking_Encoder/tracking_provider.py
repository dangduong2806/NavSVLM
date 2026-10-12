"""Read the existing trajectory source without changing its schema or vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Tracking_Encoder.trajectory_branch import TrajectorySource


class TrackingProvider:
    """Look up the current-frame objects by WAD folder/scene ID and frame ID.

    Supports the canonical ``results_botsort_top6_sorted.json`` list and the
    flat ``results_botsort_top6_sorted.jsonl`` consumed by TrajectorySource.
    A missing key returns [] so the packer can produce an all-padding sample.
    """

    def __init__(self, json_path: str | Path, *, vocabulary=None):
        self.source = TrajectorySource.from_file(str(json_path), vocabulary=vocabulary)
        self.label_to_id = self.source.label_to_id
        self.position_to_id = self.source.direction_to_id

    def get(self, scene_id: str, frame_id: int) -> list[dict[str, Any]]:
        key = (str(scene_id), int(frame_id))
        return [dict(obj) for obj in self.source.lookup.get(key, [])]

    def has_record(self, scene_id: str, frame_id: int) -> bool:
        return self.source.has_record(str(scene_id), int(frame_id))
