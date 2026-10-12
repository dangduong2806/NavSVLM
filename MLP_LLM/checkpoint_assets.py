"""Inspect checkpoint headers and prepare tracking assets without loading a model."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_header(path):
    """Read only safetensors metadata, and reject truncated/invalid payloads."""
    path = Path(path)
    with path.open("rb") as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        size = struct.unpack("<Q", first)[0]
        if not 2 <= size <= min(32 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError(f"Invalid header size (possibly an LFS pointer): {path}")
        header = json.loads(stream.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    widths = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    end = 0
    for key, value in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"]):
        start, stop = value["data_offsets"]
        shape = value["shape"]
        if value["dtype"] not in widths or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Unsupported tensor: {key}")
        if start != end or stop - start != math.prod(shape) * widths[value["dtype"]]:
            raise ValueError(f"Invalid tensor offsets: {key}")
        end = stop
    if 8 + size + end != path.stat().st_size:
        raise ValueError(f"Truncated or extra tensor data: {path}")
    return tensors


def inspect_checkpoint(directory):
    directory = Path(directory)
    tracking = tensor_header(directory / "trajectory_branch.safetensors")
    bridge = tensor_header(directory / "qformer_bridge.safetensors")
    adapter = tensor_header(directory / "adapter_model.safetensors")
    meta = read_json(directory / "trajectory_branch_config.json")
    qmeta = read_json(directory / "qformer_bridge_config.json")
    if meta.get("fusion_mode") != "concat":
        raise ValueError("Only concat checkpoints are supported")
    shape = lambda key: tracking[key]["shape"]
    prefix = "trajectory_backbone."
    layers = sorted({int(m.group(1)) for key in tracking if (m := re.match(r"trajectory_backbone\.set_encoder\.layers\.(\d+)\.", key))})
    if not layers or layers != list(range(len(layers))):
        raise ValueError("Tracking layer indices must be contiguous from zero")
    projector_key = "trajectory_token_projector.proj.2.weight" if "trajectory_token_projector.proj.2.weight" in tracking else "trajectory_token_projector.proj.1.weight"
    result = {
        "vocab_size": shape(prefix + "label_embedding.weight")[0],
        "direction_vocab_size": shape(prefix + "direction_embedding.weight")[0],
        "d_cat": shape(prefix + "label_embedding.weight")[1],
        "d_dir": shape(prefix + "direction_embedding.weight")[1],
        "d_numeric_hidden": shape(prefix + "numeric_mlp.0.weight")[0],
        "numeric_dim": shape(prefix + "numeric_mlp.0.weight")[1],
        "d_traj": shape(prefix + "slot_embedding")[2],
        "num_objects": shape(prefix + "slot_embedding")[1],
        "num_layers": len(layers),
        "ffn_dim": shape(prefix + "set_encoder.layers.0.linear1.weight")[0],
        "llm_hidden_size": shape(projector_key)[0],
        "visual_width": bridge["qformer_input_proj.1.weight"]["shape"][1],
        "qformer_memory_width": bridge["qformer_input_proj.1.weight"]["shape"][0],
        "qformer_hidden_size": bridge["qformer_to_mlp1_proj.1.weight"]["shape"][1],
        "num_query_tokens": int(qmeta["num_query_tokens"]),
    }
    if result["num_objects"] != 6 or result["numeric_dim"] != 6:
        raise ValueError("Unsupported tracking slots/numeric dimension")
    if result["num_query_tokens"] + result["num_objects"] != meta["num_image_token"]:
        raise ValueError("Checkpoint token counts disagree")
    saved_embeddings = {key: value["shape"] for key, value in adapter.items() if not (".lora_A." in key or ".lora_B." in key)}
    return {
        "architecture": result,
        "tensor_counts": {"tracking": len(tracking), "bridge": len(bridge), "adapter": len(adapter)},
        "inactive_concat_keys": [k for k in tracking if k.startswith("trajectory_cls_head.")],
        "adapter_extra_tensors": saved_embeddings,
        "adapter_config": read_json(directory / "adapter_config.json"),
        "qformer_metadata": qmeta,
        "trajectory_metadata": meta,
        "unknown_from_weights": ["tracking attention heads", "full InternVL model ID/revision", "training vocabulary provenance", "training preprocessing policy"],
    }


def notebook_labels(path):
    """Read a literal KEEP_LABELS assignment; never execute notebook cells."""
    matches = []
    for cell in read_json(path)["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "KEEP_LABELS =" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KEEP_LABELS" for t in node.targets):
                matches.append(ast.literal_eval(node.value))
    if len(matches) != 1 or not isinstance(matches[0], list) or not all(isinstance(x, str) for x in matches[0]):
        raise ValueError("Expected one literal KEEP_LABELS list in notebook")
    if len(set(matches[0])) != len(matches[0]):
        raise ValueError("Notebook labels are duplicated")
    return matches[0]


def read_flat_tracks(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    counts = Counter()
    for i, row in enumerate(rows):
        for key in ("folder_id", "frame_id", "label", "relative_position", "boxs", "movement_angle", "speed_percent"):
            if key not in row:
                raise ValueError(f"Tracking row {i} missing {key}")
        counts[str(row["folder_id"]), int(row["frame_id"])] += 1
        box = row["boxs"]
        numbers = [*box, row["movement_angle"], row["speed_percent"]]
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in numbers):
            raise ValueError(f"Invalid tracking numbers at row {i}")
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"Invalid normalized bbox at row {i}")
        if not -180 <= row["movement_angle"] <= 180 or row["speed_percent"] < 0:
            raise ValueError(f"Invalid degree/speed units at row {i}")
    if not rows or max(counts.values()) > 6:
        raise ValueError("Expected nonempty preselected tracking with <=6 objects per frame")
    return rows


def make_vocabulary(labels, directions):
    return {
        "label_to_id": {"<PAD_UNK>": 0, **{name: i for i, name in enumerate(sorted(labels), 1)}},
        "direction_to_id": {"<PAD_UNK>": 0, **{name: i for i, name in enumerate(sorted(directions), 1)}},
        "provenance": "CANDIDATE: sorted notebook KEEP_LABELS + sorted observed directions; not verified against original training source",
        "confirmed_training_mapping": False,
    }


def load_config(path):
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("checkpoint", "tracking_file", "vocabulary_file"):
        config[key] = str((path.parent / config[key]).resolve())
    base = config.get("base_model")
    if base and (base.startswith((".", "/")) or re.match(r"^[A-Za-z]:", base)):
        config["base_model"] = str((path.parent / base).resolve())
    return config
