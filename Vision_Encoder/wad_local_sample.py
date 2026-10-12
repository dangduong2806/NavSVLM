"""Load a locally extracted nine-frame WAD sample from metadata.json."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


class WADLocalSample:
    """Expose the same frame interface as WADIndexedSource for local JPEGs."""

    def __init__(self, sample_dir: str | Path):
        self.sample_dir = Path(sample_dir).resolve()
        metadata_path = self.sample_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        scene_id = str(manifest["frame_folder"])
        entries = manifest["frames"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("WAD sample metadata must contain a nonempty frames list")
        if int(manifest["num_frames"]) != len(entries):
            raise ValueError("num_frames does not match the frame list")

        index: dict[int, dict[str, str]] = {}
        for entry in entries:
            frame_id = int(entry["frame_id"])
            if frame_id in index:
                raise ValueError(f"Duplicate frame_id: {frame_id}")
            image_path = (self.sample_dir / str(entry["image_path"])).resolve()
            if not image_path.is_relative_to(self.sample_dir):
                raise ValueError(f"Image path escapes WAD sample directory: {image_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"WAD sample image is missing: {image_path}")
            index[frame_id] = {"image_path": str(image_path)}

        self.metadata = [{"frame_path": scene_id}]
        self.frame_index = {scene_id: index}

    def _load_frames(self, scene_id: str, frame_ids: list[int]) -> list[Image.Image]:
        index = self.frame_index[scene_id]
        frames = []
        for frame_id in frame_ids:
            with Image.open(index[frame_id]["image_path"]) as image:
                frames.append(image.convert("RGB"))
        return frames
