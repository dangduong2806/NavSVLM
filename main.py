"""Run the checkpoint pipeline: python main.py

Uses the notebook's 107 labels with the training code's alphabetical IDs.
Tracking attention uses four heads.
Models are retained by get_pipeline() for the lifetime of the process.
Use --video path.mp4 or --camera 0 for a continuous frame source.
Set STC_PATCH_VISION=1 to encode consecutive frames with InternViT STC-Cacher.
STC_UPDATE_TOKEN_RATIO (default 0.25) and STC_CACHE_INTERVAL (default 4)
control selective recomputation and reference refresh. STC is opt-in.
"""
import json
import math
import os
from pathlib import Path
import sys
import threading

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("HF_MODULES_CACHE", str(Path(__file__).resolve().parent / ".model_cache/modules"))

import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModel, AutoTokenizer, GenerationConfig, GenerationMixin
from peft import LoraConfig, PeftModel

from logutil import init_logger
from Tracking_Encoder.trajectory_branch import TrajectoryBackbone, TrajectoryConcatHead

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "wad_sample/images"
CHECKPOINT = ROOT / "intern-qformer-concat-1807-epoch3"
VOCABULARY_FILE = ROOT / "Tracking_Encoder/label_vocabulary.json"
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
    labels = set(json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))["labels"])
    directions = set(CLOCK_POSITIONS)
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


def load_detector():
    """Load YOLO once; the streaming runtime owns its predictor and trackers."""
    from ultralytics import YOLO
    from ultralytics.utils import ROOT as YOLO_ROOT

    config = yaml.safe_load((YOLO_ROOT / "cfg/trackers/botsort.yaml").read_text())
    config.update(with_reid=True, model="auto")
    tracker_path = CACHE_DIR / "botsort_reid.yaml"
    tracker_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    detector = YOLO(str(CACHE_DIR / "yolo11n.pt"))
    return detector, tracker_path


def track_objects(frames, label_vocab, direction_vocab, *, detector=None, tracker_path=None):
    """Advance an existing tracker and encode the final six object slots."""
    if not frames:
        raise ValueError("At least one frame is required")
    if detector is None:
        detector, tracker_path = load_detector()
    if tracker_path is None:
        raise ValueError("tracker_path is required when reusing a detector")

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
        for slot, index in enumerate(boxes.xyxy[:, 3].argsort(descending=True)[:6].tolist()):
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            track_id = int(boxes.id[index].item())
            vx, vy = map(float, states[track_id].mean[4:6])
            bearing = math.degrees(math.atan2((x1 + x2) / 2 - width / 2, width / 2))
            clock = CLOCK_POSITIONS[sum(bearing > edge for edge in (-30, -10, 10, 30))]
            label = detector.names[int(boxes.cls[index].item())]
            label = DETECTOR_LABEL_ALIASES.get(label, label)
            # Map label TEXT to the training ID; ID 0 is the training unknown ID.
            label_ids[0, slot] = label_vocab.get(label, 0)
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


def load_lora_config():
    """Read the standard LoRA settings, compatible with this project's PEFT 0.13."""
    saved = json.loads((CHECKPOINT / "adapter_config.json").read_text())
    # PEFT 0.18 added optional fields that are null/false in this checkpoint.
    # Construct the equivalent ordinary LoRA config without editing the original file.
    return LoraConfig(
        r=saved["r"], lora_alpha=saved["lora_alpha"],
        lora_dropout=saved["lora_dropout"], target_modules=saved["target_modules"],
        bias=saved["bias"], task_type=saved["task_type"], inference_mode=True,
        base_model_name_or_path=saved["base_model_name_or_path"],
    )


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

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # AutoModel loads InternLM2 support from the official base repository.
    # The local demo's InternVLChatModel only implements Llama/Qwen2.
    model = AutoModel.from_pretrained(
        BASE_MODEL, trust_remote_code=True, torch_dtype=DTYPE,
        low_cpu_mem_usage=True, use_flash_attn=False, cache_dir=str(CACHE_DIR),
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), trust_remote_code=True, use_fast=False)
    # Restore generation before PEFT captures the language model's methods.
    model.language_model = enable_internlm_generation(model.language_model)
    model.language_model = PeftModel.from_pretrained(
        model.language_model, str(CHECKPOINT), config=load_lora_config(), is_trainable=False,
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
    stc_enabled = os.environ.get("STC_PATCH_VISION", "0")
    if stc_enabled not in ("0", "1"):
        raise ValueError("STC_PATCH_VISION must be 0 or 1")
    if stc_enabled == "1":
        from Vision_Encoder.stc_cacher import enable_stc_cacher

        enable_stc_cacher(
            model.vision_model,
            update_token_ratio=float(os.environ.get("STC_UPDATE_TOKEN_RATIO", "0.25")),
            cache_interval=int(os.environ.get("STC_CACHE_INTERVAL", "4")),
        )
    return model, tokenizer, qformer, queries, qtokenizer, projections.to(DEVICE).eval()


@torch.inference_mode()
def generate_guidance(model_bundle, tracking_bundle, visual_tokens, tracking_inputs):
    """Generate from the latest frame using already loaded model instances."""
    from MLP_LLM.conversation import get_conv_template
    model, tokenizer, qformer, queries, qtokenizer, projections = model_bundle
    tracking_encoder, trajectory_projection = tracking_bundle
    label_ids, direction_ids, numeric_features, object_mask = tracking_inputs

    # 4. PROMPT-AWARE Q-FORMER: the checkpoint configuration requires text here.
    visual_input = projections["qformer_input_proj"](visual_tokens.float()).to(DTYPE)
    qtext = qtokenizer(PROMPT, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    query_mask = torch.ones((1, 32), dtype=torch.long, device=DEVICE)
    compressed = qformer(
        input_ids=qtext.input_ids, query_embeds=queries,
        attention_mask=torch.cat([query_mask, qtext.attention_mask], dim=1),
        encoder_hidden_states=visual_input,
        encoder_attention_mask=torch.ones((1, 256), dtype=torch.long, device=DEVICE),
        return_dict=True,
    ).last_hidden_state[:, :32]
    visual_tokens = projections["qformer_to_mlp1_proj"](compressed.float())
    visual_tokens = model.mlp1(visual_tokens.to(DTYPE))  # [1,32,2048]

    # 5. TRACKING ENCODER: [1,6,384] -> trained projector -> [1,6,2048].
    trajectory_tokens = tracking_encoder(label_ids, direction_ids, numeric_features, object_mask)
    trajectory_tokens = trajectory_projection(trajectory_tokens) * object_mask.unsqueeze(-1)

    # 6. CHECKPOINT FUSION: concatenate AFTER the visual MLP, matching training.
    # Both branches already have the LLM width: [1,32,2048] + [1,6,2048].
    fused_tokens = torch.cat([visual_tokens, trajectory_tokens.to(DTYPE)], dim=1)

    # 7. LLM INPUT: use the InternLM conversation template and checkpoint tokenizer.
    template = get_conv_template(model.template)
    template.system_message = "You are a navigation assistant for visually impaired users."
    template.append_message(template.roles[0], "<img>" + "<IMG_CONTEXT>" * 38 + "</img>\n" + PROMPT)
    template.append_message(template.roles[1], None)
    text = tokenizer(template.get_prompt(), return_tensors="pt").to(DEVICE)
    embeddings = model.language_model.get_input_embeddings()(text.input_ids)
    image_positions = text.input_ids == tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    embeddings[image_positions] = fused_tokens.reshape(38, 2048).to(embeddings)
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
    return answer


_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    """One lazy runtime per process, reused by the CLI and persistent worker."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            from streaming_pipeline import StreamingPipeline

            _pipeline = StreamingPipeline(backend=sys.modules[__name__])
    return _pipeline


def main(images_dir=None, limit=9):
    answer = get_pipeline().run_directory(images_dir or IMAGES_DIR, limit=limit)
    print("Generated text:", answer, flush=True)
    return answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images-dir", type=Path, default=None)
    source.add_argument("--video", type=Path, help="Stream a local video file")
    source.add_argument("--camera", type=int, help="Stream an OpenCV camera index, e.g. 0")
    parser.add_argument("--limit", type=int, default=9, help="Directory frame limit; 0 means all JPGs")
    parser.add_argument("--guidance-every", type=int, default=9, help="Generate once per N video/camera frames")
    args = parser.parse_args()
    if args.limit < 0 or args.guidance_every < 1:
        parser.error("--limit must be nonnegative and --guidance-every must be positive")
    try:
        if args.video is not None or args.camera is not None:
            get_pipeline().run_video(str(args.video) if args.video is not None else args.camera,
                                     guidance_every=args.guidance_every)
        else:
            main(args.images_dir, args.limit)
    except KeyboardInterrupt:
        print("Streaming stopped.", flush=True)
