"""Report readiness separately from successful checkpoint tensor loading."""
import argparse
import json
from importlib.util import find_spec

from MLP_LLM.checkpoint_assets import inspect_checkpoint, load_config, read_json, read_flat_tracks, write_json


def validate(config, *, scene_id=None, frame_id=8, verify_weights=False):
    errors, warnings = [], []
    report = {"ready_for_inference": False, "errors": errors, "warnings": warnings}
    missing = [name for name in ("torch", "transformers", "huggingface_hub", "peft", "accelerate", "safetensors", "sentencepiece", "einops", "timm") if find_spec(name) is None]
    if missing:
        errors.append("Missing runtime dependencies: " + ", ".join(missing) + "; install requirements-pipeline.txt")
    try:
        inspection = inspect_checkpoint(config["checkpoint"])
        report["checkpoint"] = inspection
        arch = inspection["architecture"]
        if config.get("architecture") != arch:
            errors.append("YAML architecture differs from checkpoint tensor dimensions; regenerate or correct it")
        vocab = read_json(config["vocabulary_file"])
        for name, expected in (("label_to_id", arch["vocab_size"]), ("direction_to_id", arch["direction_vocab_size"])):
            mapping = vocab[name]
            if len(mapping) != expected or mapping.get("<PAD_UNK>") != 0 or any(type(x) is not int for x in mapping.values()) or sorted(mapping.values()) != list(range(expected)):
                errors.append(f"Invalid {name}; expected {expected} contiguous IDs with PAD=0")
        if vocab.get("confirmed_training_mapping") is not True:
            errors.append("Vocabulary is a candidate; confirm ID order against training source before semantic inference")
        rows = read_flat_tracks(config["tracking_file"])
        if any(row["label"] not in vocab["label_to_id"] or row["relative_position"] not in vocab["direction_to_id"] for row in rows):
            errors.append("Tracking contains objects outside fixed vocabulary")
        frames = sorted({(row["folder_id"], row["frame_id"]) for row in rows})
        report["available_scene_frames"] = frames
        report["tracking_rows"] = len(rows)
        if scene_id is not None and (scene_id, frame_id) not in frames:
            errors.append(f"No real tracking record for scene={scene_id}, frame={frame_id}; provide matching images/tracks")
        if not config.get("base_model"):
            errors.append("base_model is missing: adapter base_model_name_or_path identifies the LLM, not the full InternVL")
        heads = config.get("tracking_num_heads")
        if type(heads) is not int or heads <= 0 or arch["d_traj"] % heads:
            errors.append("tracking_num_heads must be supplied from training config and divide d_traj; not inferable from tensor shapes")
        if config.get("qformer_source") != inspection["qformer_metadata"]["source_model"]:
            errors.append("Q-Former source differs from checkpoint metadata")
        pp = config.get("preprocess", {})
        if pp.get("confirmed_training_policy") is not True:
            errors.append("Preprocess/tiling policy needs confirmation for this checkpoint")
        if type(pp.get("image_size")) is not int or pp["image_size"] <= 0 or type(pp.get("max_tiles")) is not int or pp["max_tiles"] < 1 or type(pp.get("use_thumbnail")) is not bool:
            errors.append("Invalid preprocess dimensions/thumbnail setting")
        if not config.get("base_revision") or not config.get("qformer_revision"):
            warnings.append("Source revisions are not pinned; save exact resolved revisions with inference output")
        if inspection["inactive_concat_keys"]:
            warnings.append("Checkpoint also contains CLS head tensors; concat mode leaves this inactive head unused")
        runtime = config.get("runtime", {})
        if runtime.get("dtype") not in ("float32", "float16", "bfloat16"):
            errors.append("Unsupported runtime dtype")
        try:
            import torch
            report["runtime"] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
            device = torch.device(runtime.get("device", "cpu"))
            if device.type not in ("cpu", "cuda"):
                errors.append("Only cpu/cuda runtime is supported")
            if device.type == "cuda" and not torch.cuda.is_available():
                errors.append("CUDA requested but unavailable in this Python runtime")
            if device.type == "cpu" and (runtime.get("load_in_4bit") or runtime.get("dtype") != "float32"):
                errors.append("CPU path requires float32 and load_in_4bit=false")
        except ImportError:
            errors.append("PyTorch is not installed in this interpreter")
        if verify_weights:
            from MLP_LLM.checkpoint_loading import verify_checkpoint_tensors
            report["weight_verification"] = verify_checkpoint_tensors(config["checkpoint"], arch)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    report["ready_for_inference"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-dir")
    parser.add_argument("--frame-id", type=int, default=8)
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--output", default="output/assets_report.json")
    args = parser.parse_args()
    scene = None
    if args.sample_dir:
        from pathlib import Path
        scene = read_json(Path(args.sample_dir) / "metadata.json")["frame_folder"]
    report = validate(load_config(args.config), scene_id=scene, frame_id=args.frame_id, verify_weights=args.verify_weights)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["ready_for_inference"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
