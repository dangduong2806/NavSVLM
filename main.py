"""Run the checkpoint pipeline: python main.py

Uses the notebook's 107 labels with the training code's alphabetical IDs.
Tracking attention uses four heads.
"""
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
from transformers import AutoModel, AutoTokenizer
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


def track_objects(frames, label_vocab, direction_vocab):
    """Run BoT-SORT/ReID on nine images and encode the final six object slots."""
    from ultralytics import YOLO
    from ultralytics.utils import ROOT as YOLO_ROOT

    config = yaml.safe_load((YOLO_ROOT / "cfg/trackers/botsort.yaml").read_text())
    config.update(with_reid=True, model="auto")
    tracker_path = CACHE_DIR / "botsort_reid.yaml"
    tracker_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    detector = YOLO(str(CACHE_DIR / "yolo11n.pt"))

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


def load_models():
    """Load the pretrained InternVL base, language adapter, and Q-Former bridge."""
    from Concat_Q_Former.qformer_bridge import _load_qformer_from_source

    # AutoModel loads InternLM2 support from the official base repository.
    # The local demo's InternVLChatModel only implements Llama/Qwen2.
    model = AutoModel.from_pretrained(
        BASE_MODEL, trust_remote_code=True, torch_dtype=DTYPE,
        low_cpu_mem_usage=True, use_flash_attn=False, cache_dir=str(CACHE_DIR),
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT), trust_remote_code=True, use_fast=False)
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
    return model, tokenizer, qformer, queries, qtokenizer, projections.to(DEVICE).eval()


@torch.inference_mode()
def main():
    (CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    init_logger(str(CACHE_DIR) + os.sep)

    # Load the full label vocabulary before downloading the large base weights.
    label_vocab, direction_vocab = load_vocabularies()

    from Concat_Q_Former.qformer_bridge import _extract_vit_tokens
    from MLP_LLM.conversation import get_conv_template

    # 1. INPUTS: the sample supplies nine images; vision uses only the last one.
    frames = []
    for path in sorted(IMAGES_DIR.glob("*.jpg"))[:9]:
        with Image.open(path) as image:
            frames.append(np.array(image.convert("RGB")))

    # 2. TRACKING: keep the online tracker; use training vocabulary IDs.
    label_ids, direction_ids, numeric_features, object_mask = track_objects(frames, label_vocab, direction_vocab)
    tracking_encoder, trajectory_projection = load_tracking_encoder(TRACKING_HEADS)
    model, tokenizer, qformer, queries, qtokenizer, projections = load_models()

    # 3. VISION: 448px RGB -> normalized pixels -> 256 tokens of width 4096.
    image = Image.fromarray(frames[-1]).resize((448, 448), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    visual_tokens = _extract_vit_tokens(model, ((pixels - mean) / std).to(DTYPE))

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
    print("Generated text:", answer)


if __name__ == "__main__":
    main()
