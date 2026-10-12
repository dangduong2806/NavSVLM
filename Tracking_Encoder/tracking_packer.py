"""Convert TrajectorySource objects to the existing TrajectoryBackbone inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


NUMERIC_FIELDS = ("x1", "y1", "x2", "y2", "movement_angle", "speed_percent")
MAX_OBJECTS = 6


def pack_tracking_objects(
    objects: Sequence[Mapping],
    *,
    label_to_id: Mapping[str, int],
    position_to_id: Mapping[str, int],
    max_objects: int = MAX_OBJECTS,
) -> dict[str, torch.Tensor]:
    """Pack one scene to tensors with a leading batch dimension of one.

    Input objects must already use TrajectorySource's canonical units:
    normalized bbox, movement_angle in degrees/180, speed_percent in percent.
    Object order is preserved; selection of the top six occurs upstream.
    Unknown labels/positions are rejected to expose a vocabulary mismatch.
    """
    if max_objects != MAX_OBJECTS:
        raise ValueError("The existing TrajectoryBackbone contract fixes max_objects=6")
    if len(objects) > max_objects:
        raise ValueError("Expected at most six preselected objects")

    category_ids = torch.zeros((1, max_objects), dtype=torch.long)
    position_ids = torch.zeros((1, max_objects), dtype=torch.long)
    numeric_features = torch.zeros((1, max_objects, len(NUMERIC_FIELDS)), dtype=torch.float32)
    mask = torch.zeros((1, max_objects), dtype=torch.long)

    for slot, obj in enumerate(objects):
        label = str(obj["label"])
        position = str(obj["relative_position"])
        if label not in label_to_id or position not in position_to_id:
            raise ValueError(f"Object #{slot} is outside the checkpoint vocabulary: {label!r}, {position!r}")
        values = torch.tensor([float(obj[name]) for name in NUMERIC_FIELDS], dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError(f"Object #{slot} has a non-finite numeric feature")
        x1, y1, x2, y2, angle, speed = values.tolist()
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Object #{slot} has an invalid normalized bounding box")
        if not (-1 <= angle <= 1) or speed < 0:
            raise ValueError(f"Object #{slot} has an invalid angle or speed unit")

        category_ids[0, slot] = label_to_id[label]
        position_ids[0, slot] = position_to_id[position]
        numeric_features[0, slot] = values
        mask[0, slot] = 1

    return {
        "category_ids": category_ids,
        "position_ids": position_ids,
        "numeric_features": numeric_features,
        "mask": mask,
    }


def tracking_encoder_kwargs(packed: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate Phase 0 names to TrajectoryBackbone.forward parameter names."""
    return {
        "label_ids": packed["category_ids"],
        "direction_ids": packed["position_ids"],
        "numeric_feats": packed["numeric_features"],
        "object_mask": packed["mask"],
    }
