"""Run the checkpoint pipeline: python main.py

Uses the notebook's 107 labels with the training code's alphabetical IDs.
Tracking attention uses four heads.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("HF_MODULES_CACHE", str(Path(__file__).resolve().parent / ".model_cache/modules"))

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, GenerationConfig, GenerationMixin
from peft import PeftModel

from logutil import init_logger
from Tracking_Encoder.trajectory_branch import TrajectoryBackbone, TrajectoryConcatHead

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "wad_sample4/images"
CHECKPOINT = ROOT / "intern-qformer-concat-1807-epoch3"
VOCABULARY_FILE = CHECKPOINT / "inference/vocabulary.candidate.json"
CACHE_DIR = ROOT / ".model_cache"
# This base matches the adapter's InternLM2.5 name and the saved vision/LLM widths.
# Confirm against the training config if a different full InternVL base was used.
BASE_MODEL = "OpenGVLab/InternVL2_5-2B"
TRACKING_HEADS = 4  # Four attention heads per tracking Transformer layer.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
PROMPT = (
    "Describe the scene for a visually impaired user based on the final frame.\n"
    "Focus on immediate obstacles, safe direction, and what action the user should take.\n"
    "Provide only the final spoken guidance in natural language."
)
CLOCK_POSITIONS = ["10 o'clock", "11 o'clock", "12 o'clock", "1 o'clock", "2 o'clock"]
# COCO detector names that differ from the notebook's category names.
DETECTOR_LABEL_ALIASES = {
    "car": "car_(automobile)",
    "bus": "bus_(vehicle)",
    "train": "train_(railroad_vehicle)",
    "traffic light": "traffic_light",
    "fire hydrant": "fireplug",
    "stop sign": "stop_sign",
    "parking meter": "parking_meter",
    "dining table": "dining_table",
    "tv": "television_set",
    "sports ball": "ball",
}


def load_vocabularies():
    """Assign IDs using the notebook's full label set and training's sorting rule."""
    # The sample TXT contains only observed classes, so cannot define all 107 IDs.
    # Assumes the notebook's KEEP_LABELS is the same set used during training.
    candidate = json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))
    labels = set(candidate["label_to_id"]) - {"<PAD_UNK>"}
    directions = set(candidate["direction_to_id"]) - {"<PAD_UNK>"}
    with safe_open(str(CHECKPOINT / "trajectory_branch.safetensors"), framework="pt") as weights:
        label_count = weights.get_slice("trajectory_backbone.label_embedding.weight").get_shape()[0] - 1
        direction_count = weights.get_slice("trajectory_backbone.direction_embedding.weight").get_shape()[0] - 1
    # A partial vocabulary would load valid tensor indices with the WRONG meanings.
    if len(labels) != label_count or len(directions) != direction_count:
        raise ValueError(
            f"{VOCABULARY_FILE.name} contains {len(labels)} labels and {len(directions)} directions; "
            f"the checkpoint needs {label_count} labels and {direction_count} directions. "
            "Use the complete training vocabulary. Do not use YOLO class IDs."
        )
    label_vocab = {"<PAD_UNK>": 0, **{name: i for i, name in enumerate(sorted(labels), 1)}}
    direction_vocab = {"<PAD_UNK>": 0, **{name: i for i, name in enumerate(sorted(directions), 1)}}
    return label_vocab, direction_vocab


def track_objects(frames, label_vocab, direction_vocab, *, diagnostics=False):
    """Run BoT-SORT/ReID on nine images and encode the final six object slots."""
    from ultralytics import YOLO
    from ultralytics.utils import ROOT as YOLO_ROOT

    config = yaml.safe_load((YOLO_ROOT / "cfg/trackers/botsort.yaml").read_text())
    config.update(with_reid=True, model="auto")
    tracker_path = CACHE_DIR / "botsort_reid.yaml"
    tracker_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    detector_weights = CACHE_DIR / "yolo11n.pt"
    detector = YOLO(str(detector_weights if detector_weights.is_file() else "yolo11n.pt"))

    for frame in frames:
        result = detector.track(
            np.ascontiguousarray(frame[:, :, ::-1]), persist=True,
            tracker=str(tracker_path), conf=0.1, device=DEVICE, verbose=False,
        )[0]

    label_ids = torch.zeros((1, 6), dtype=torch.long, device=DEVICE)
    direction_ids = torch.zeros_like(label_ids)
    numeric_features = torch.zeros((1, 6, 6), dtype=torch.float32, device=DEVICE)
    object_mask = torch.zeros_like(label_ids)
    height, width = frames[-1].shape[:2]
    boxes = result.boxes
    if boxes is not None and boxes.is_track:
        states = {int(t.track_id): t for t in detector.predictor.trackers[0].tracked_stracks}
        # The supplied example orders objects by descending box bottom position.
        selected = []
        for index in boxes.xyxy[:, 3].argsort(descending=True).tolist():
            label = detector.names[int(boxes.cls[index].item())]
            label = DETECTOR_LABEL_ALIASES.get(label, label)
            if label in label_vocab and int(boxes.id[index].item()) in states:
                selected.append((index, label))
            if len(selected) == 6:
                break
        if diagnostics:
            print("Tracked objects (nearest box-bottom first):")
        for slot, (index, label) in enumerate(selected):
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            track_id = int(boxes.id[index].item())
            vx, vy = map(float, states[track_id].mean[4:6])
            bearing = math.degrees(math.atan2((x1 + x2) / 2 - width / 2, width / 2))
            clock = CLOCK_POSITIONS[sum(bearing > edge for edge in (-30, -10, 10, 30))]
            if diagnostics:
                print(f"  {slot + 1}. id={track_id} {label} {clock} "
                      f"confidence={float(boxes.conf[index]):.2f} "
                      f"box={[round(v, 1) for v in (x1, y1, x2, y2)]}")
            label_ids[0, slot] = label_vocab[label]
            direction_ids[0, slot] = direction_vocab[clock]
            numeric_features[0, slot] = torch.tensor([
                x1 / width, y1 / height, x2 / width, y2 / height,
                math.atan2(vy, vx) / math.pi,
                100 * math.hypot(vx, vy) / math.hypot(width, height),
            ], device=DEVICE)
            object_mask[0, slot] = 1
    # The TXT stores angles in degrees; live tracking above already divides by pi.
    return label_ids, direction_ids, numeric_features, object_mask


def load_tracking_encoder(num_heads):
    """Restore the four-layer, 384-wide tracking branch and its 2048-wide head."""
    state = load_file(str(CHECKPOINT / "trajectory_branch.safetensors"))
    encoder = TrajectoryBackbone(
        vocab_size=108, direction_vocab_size=6, d_traj=384,
        num_layers=4, num_heads=num_heads, ffn_dim=768,
    )
    projector = TrajectoryConcatHead(input_dim=384, output_dim=2048)
    # This checkpoint predates the Dropout entries in the local Sequential modules.
    # Removing dropout gives the exact saved .0/.2 and .0/.1 parameter names.
    encoder.numeric_mlp = nn.Sequential(*[m for m in encoder.numeric_mlp if not isinstance(m, nn.Dropout)])
    encoder.object_mlp = nn.Sequential(*[m for m in encoder.object_mlp if not isinstance(m, nn.Dropout)])
    projector.proj = nn.Sequential(*[m for m in projector.proj if not isinstance(m, nn.Dropout)])
    encoder.load_state_dict({
        k.removeprefix("trajectory_backbone."): v
        for k, v in state.items() if k.startswith("trajectory_backbone.")
    }, strict=True)
    projector.load_state_dict({
        k.removeprefix("trajectory_token_projector."): v
        for k, v in state.items() if k.startswith("trajectory_token_projector.")
    }, strict=True)
    # trajectory_cls_head.* is also saved, but is unused in the configured concat mode.
    return encoder.to(DEVICE).eval(), projector.to(DEVICE).eval()


def enable_internlm_generation(language_model):
    """Restore generation for legacy InternLM2 under newer Transformers."""
    if not isinstance(language_model, GenerationMixin):
        # New Transformers no longer supplies generate() through PreTrainedModel.
        # Add it to this instance without reloading weights or editing cached code.
        class InternLM2WithGeneration(type(language_model), GenerationMixin):
            @classmethod
            def _supports_default_dynamic_cache(cls):
                # InternLM2's forward/prepare methods expect legacy tuple caches.
                return False

        language_model.__class__ = InternLM2WithGeneration
    if language_model.generation_config is None:
        language_model.generation_config = GenerationConfig.from_model_config(language_model.config)
    return language_model


def load_models():
    """Load the pretrained InternVL base, language adapter, and Q-Former bridge."""
    from Concat_Q_Former.qformer_bridge import _load_qformer_from_source

    # AutoModel loads InternLM2 support from the official base repository.
    # The local demo's InternVLChatModel only implements Llama/Qwen2.
    model = AutoModel.from_pretrained(
        BASE_MODEL, trust_remote_code=True, dtype=DTYPE,
        low_cpu_mem_usage=True, use_flash_attn=False, cache_dir=str(CACHE_DIR),
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), trust_remote_code=True, use_fast=False)
    # Restore generation before PEFT captures the language model's methods.
    model.language_model = enable_internlm_generation(model.language_model)
    model.language_model = PeftModel.from_pretrained(
        model.language_model, str(CHECKPOINT), is_trainable=False,
    ).eval()

    config = json.loads((CHECKPOINT / "qformer_bridge_config.json").read_text())
    qformer, queries, qtokenizer, _ = _load_qformer_from_source(config["source_model"], str(CACHE_DIR))
    qformer = qformer.to(device=DEVICE, dtype=DTYPE).eval()
    queries = queries[:, :config["num_query_tokens"]].to(device=DEVICE, dtype=DTYPE)
    # Saved bridge: ViT width 4096 -> cross-attention width 1408;
    # Q-Former query width 768 -> pre-MLP width 4096.
    projections = nn.ModuleDict({
        "qformer_input_proj": nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 1408)),
        "qformer_to_mlp1_proj": nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 4096)),
    })
    projections.load_state_dict(load_file(str(CHECKPOINT / "qformer_bridge.safetensors")), strict=True)
    return model, tokenizer, qformer, queries, qtokenizer, projections.to(DEVICE).eval()


def resolve_image_paths(images_dir: Path, image_file: Path | None):
    """Select one image or all frames in a folder, in filename order."""
    if image_file is not None:
        if not image_file.is_file() or image_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise FileNotFoundError(f"Image not found or unsupported: {image_file}")
        return [image_file]
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    paths = sorted(path for path in images_dir.iterdir()
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No JPEG/PNG images in {images_dir}")
    return paths


def check_local_assets(image_paths, *, use_tracking):
    """Cheap checks before downloading the base and Q-Former source."""
    load_vocabularies()
    required = [CHECKPOINT / name for name in (
        "adapter_model.safetensors", "adapter_config.json", "qformer_bridge.safetensors",
        "qformer_bridge_config.json", "trajectory_branch.safetensors",
        "inference/vocabulary.candidate.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if use_tracking and len(image_paths) != 9:
        missing.append(f"Tracking requires exactly nine ordered frames; found {len(image_paths)}")
    if missing:
        raise FileNotFoundError("\n".join(missing))
    print(f"Local assets OK: {len(image_paths)} image(s), checkpoint tensors, 107 labels")
    print(f"Input: {image_paths[0]}" + (f" ... {image_paths[-1]}" if len(image_paths) > 1 else ""))
    print(f"Runtime: {DEVICE}, {DTYPE}; training mapping and four-head assumption remain unverified")


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run NavVLM on an image or a nine-frame scene")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--images-dir", type=Path, default=IMAGES_DIR,
                             help="Folder of nine ordered JPEG/PNG frames (default: %(default)s)")
    input_group.add_argument("--image", type=Path,
                             help="Single JPEG/PNG image; runs without tracking")
    parser.add_argument("--check-only", action="store_true", help="Validate local assets without model downloads")
    parser.add_argument("--skip-tracking", action="store_true", help="Use empty object slots when no detector is available; image demo only")
    parser.add_argument("--diagnostics", action="store_true", help="Print selected tracks and multimodal input fingerprints")
    parser.add_argument(
    "--ablation",
    choices=["full", "vision_only", "tracking_only", "all_zero"],
    default="full",
    help="Ablation test for visual/tracking multimodal tokens",
    )
    args = parser.parse_args(argv)
    (CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    init_logger(str(CACHE_DIR) + os.sep)

    # Load the full label vocabulary before downloading the large base weights.
    label_vocab, direction_vocab = load_vocabularies()
    image_paths = resolve_image_paths(args.images_dir, args.image)
    use_tracking = not args.skip_tracking and args.image is None
    check_local_assets(image_paths, use_tracking=use_tracking)
    if args.check_only:
        return 0

    from Concat_Q_Former.qformer_bridge import _extract_vit_tokens
    from MLP_LLM.conversation import get_conv_template

    # 1. INPUTS: tracking uses nine frames; vision uses the final image.
    frames = []
    for path in image_paths:
        with Image.open(path) as image:
            frames.append(np.array(image.convert("RGB")))

    # 2. TRACKING: keep the online tracker; use training vocabulary IDs.
    if not use_tracking:
        label_ids = torch.zeros(1, 6, dtype=torch.long, device=DEVICE)
        direction_ids = torch.zeros_like(label_ids)
        numeric_features = torch.zeros(1, 6, 6, device=DEVICE)
        object_mask = torch.zeros_like(label_ids)
        print("Image demo: tracking slots are empty; no BoT-SORT result was used")
    else:
        label_ids, direction_ids, numeric_features, object_mask = track_objects(
            frames, label_vocab, direction_vocab, diagnostics=args.diagnostics)
    tracking_encoder, trajectory_projection = load_tracking_encoder(TRACKING_HEADS)
    model, tokenizer, qformer, queries, qtokenizer, projections = load_models()

    # 3. VISION: 448px RGB -> normalized pixels -> 256 tokens of width 4096.
    from Vision_Encoder.preprocess import image_tiles
    pixels = image_tiles(Image.fromarray(frames[-1]), image_size=448, max_tiles=6, use_thumbnail=True).to(DEVICE, dtype=DTYPE)
    tile_count = pixels.shape[0]
    visual_tokens = _extract_vit_tokens(model, pixels)

    # 4. PROMPT-AWARE Q-FORMER: the checkpoint configuration requires text here.
    visual_input = projections["qformer_input_proj"](visual_tokens.float()).to(DTYPE)
    qtext = qtokenizer([PROMPT] * tile_count, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    query_count = queries.shape[1]
    query_mask = torch.ones((tile_count, query_count), dtype=torch.long, device=DEVICE)
    compressed = qformer(
        input_ids=qtext.input_ids, query_embeds=queries.expand(tile_count, -1, -1),
        attention_mask=torch.cat([query_mask, qtext.attention_mask], dim=1),
        encoder_hidden_states=visual_input,
        encoder_attention_mask=torch.ones((tile_count, visual_input.shape[1]), dtype=torch.long, device=DEVICE),
        return_dict=True,
    ).last_hidden_state[:, :query_count]
    visual_tokens = projections["qformer_to_mlp1_proj"](compressed.float())
    visual_tokens = model.mlp1(visual_tokens.to(DTYPE))  # [1,32,2048]

    # 5. TRACKING ENCODER: [1,6,384] -> trained projector -> [1,6,2048].
    trajectory_tokens = tracking_encoder(label_ids.repeat(tile_count, 1), direction_ids.repeat(tile_count, 1),
                                         numeric_features.repeat(tile_count, 1, 1), object_mask.repeat(tile_count, 1))
    trajectory_tokens = trajectory_projection(trajectory_tokens) * object_mask.repeat(tile_count, 1).unsqueeze(-1)

    # 6. CHECKPOINT FUSION: concatenate AFTER the visual MLP, matching training.
    # Both branches already have the LLM width: [1,32,2048] + [1,6,2048].
    # 6. ABLATION + FUSION
    visual_for_fusion = visual_tokens
    trajectory_for_fusion = trajectory_tokens.to(DTYPE)

    if args.ablation == "vision_only":
        # Chỉ giữ thông tin từ ảnh
        trajectory_for_fusion = torch.zeros_like(trajectory_for_fusion)

    elif args.ablation == "tracking_only":
        # Chỉ giữ thông tin tracking
        visual_for_fusion = torch.zeros_like(visual_for_fusion)

    elif args.ablation == "all_zero":
        # Xóa cả vision lẫn tracking
        visual_for_fusion = torch.zeros_like(visual_for_fusion)
        trajectory_for_fusion = torch.zeros_like(trajectory_for_fusion)

    fused_tokens = torch.cat(
        [visual_for_fusion, trajectory_for_fusion],
        dim=1
    )

    print(f"Ablation mode: {args.ablation}")
    if args.diagnostics:
        fingerprint = lambda tensor: hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
        print(
            f"Input fingerprints: "
            f"final_frame={hashlib.sha256(frames[-1].tobytes()).hexdigest()[:16]} "
            f"visual_tokens={fingerprint(visual_tokens)} "
            f"trajectory_tokens={fingerprint(trajectory_tokens)} "
            f"fused_tokens={fingerprint(fused_tokens)}"
        )
    # 7. LLM INPUT: use the InternLM conversation template and checkpoint tokenizer.
    template = get_conv_template(model.template)
    template.system_message = "You are a navigation assistant for visually impaired users."
    template.append_message(template.roles[0], "<img>" + "<IMG_CONTEXT>" * (38 * tile_count) + "</img>\n" + PROMPT)
    template.append_message(template.roles[1], None)
    text = tokenizer(template.get_prompt(), return_tensors="pt").to(DEVICE)
    embeddings = model.language_model.get_input_embeddings()(text.input_ids)
    image_positions = text.input_ids == tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if int(image_positions.sum()) != 38 * tile_count or fused_tokens.shape != (tile_count, 38, 2048):
        raise ValueError("Multimodal placeholder/fusion mismatch")
    embeddings[image_positions] = fused_tokens.reshape(38 * tile_count, 2048).to(embeddings)
    # Match the supplied training/inference bridge: zero padded features remain
    # placeholder positions in the LLM; padding is masked inside the set encoder.

    # 8. OUTPUT: the language model now runs with the checkpoint's LoRA adapter.
    stop_id = tokenizer.convert_tokens_to_ids(template.sep)
    output_ids = model.language_model.generate(
        inputs_embeds=embeddings, attention_mask=text.attention_mask,
        max_new_tokens=128, do_sample=False, use_cache=True,
        eos_token_id=stop_id, pad_token_id=stop_id,
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).split(template.sep)[0].strip()
    print(f"Image tiles: {tile_count}; multimodal tokens: {38 * tile_count}; tracked objects: {int(object_mask.sum())}")
    print("Generated text:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
