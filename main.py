"""A simple example of the pipeline in system.md. Run: python main.py

YOLO uses pretrained weights. The other networks are small, untrained examples
so the data flow is easy to run and read; their output is not trained guidance.
"""
import math
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    InstructBlipQFormerConfig,
    InstructBlipQFormerModel,
    PreTrainedTokenizerFast,
)

from logutil import init_logger
from Tracking_Encoder.trajectory_branch import TrajectoryBackbone, TrajectoryConcatHead

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "wad_sample/images"
CACHE_DIR = ROOT / ".model_cache"
PROMPT = "Describe the scene for a visually impaired user. Focus on obstacles and walking direction."
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def track_objects(frames):
    """Track the sequence and turn the final detections into six object slots."""
    from ultralytics import YOLO
    from ultralytics.utils import ROOT as YOLO_ROOT

    # Enable appearance features (ReID) in BoT-SORT.
    config = yaml.safe_load((YOLO_ROOT / "cfg/trackers/botsort.yaml").read_text())
    config.update(with_reid=True, model="auto")
    tracker_path = CACHE_DIR / "botsort_reid.yaml"
    tracker_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    detector = YOLO(str(CACHE_DIR / "yolo11n.pt"))

    # persist=True keeps object identities and motion state across all nine frames.
    for frame in frames:
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        result = detector.track(
            bgr, persist=True, tracker=str(tracker_path),
            conf=0.1, device=DEVICE, verbose=False,
        )[0]

    # Zero slots represent padding when fewer than six objects are available.
    label_ids = torch.zeros((1, 6), dtype=torch.long)
    direction_ids = torch.zeros((1, 6), dtype=torch.long)
    numeric_features = torch.zeros((1, 6, 6), dtype=torch.float32)
    object_mask = torch.zeros((1, 6), dtype=torch.long)
    boxes = result.boxes
    height, width = frames[-1].shape[:2]

    if boxes is not None and boxes.is_track:
        states = {int(t.track_id): t for t in detector.predictor.trackers[0].tracked_stracks}
        # Keep the six highest-confidence tracks visible in the final frame.
        for slot, index in enumerate(boxes.conf.argsort(descending=True)[:6].tolist()):
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            track_id = int(boxes.id[index].item())
            vx, vy = map(float, states[track_id].mean[4:6])

            # With a 90-degree horizontal FOV, focal length is width/2.
            # Direction IDs 1..5 correspond to 10, 11, 12, 1, and 2 o'clock.
            angle = math.degrees(math.atan2((x1 + x2) / 2 - width / 2, width / 2))
            label_ids[0, slot] = int(boxes.cls[index].item()) + 1
            direction_ids[0, slot] = 1 + sum(angle > edge for edge in (-30, -10, 10, 30))
            numeric_features[0, slot] = torch.tensor([
                x1 / width, y1 / height, x2 / width, y2 / height,
                math.atan2(vy, vx) / math.pi,
                100 * math.hypot(vx, vy) / math.hypot(width, height),
            ])
            object_mask[0, slot] = 1

    # Numeric order: normalized xyxy, movement angle/pi, diagonal % per frame.
    return (
        label_ids.to(DEVICE), direction_ids.to(DEVICE),
        numeric_features.to(DEVICE), object_mask.to(DEVICE),
        len(detector.names) + 1,
    )


def make_tokenizer():
    """Create a small local vocabulary for the untrained language-model example."""
    special = ["[PAD]", "[UNK]", "[EOS]", "<IMG_CONTEXT>"]
    splitter = Whitespace()
    words = [word for word, _ in splitter.pre_tokenize_str(PROMPT)]
    vocab = {word: i for i, word in enumerate(dict.fromkeys(special + words))}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = splitter
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
        eos_token="[EOS]", additional_special_tokens=["<IMG_CONTEXT>"],
    )


def make_models(tokenizer):
    """Use the project's model classes with small dimensions and random weights."""
    from MLP_LLM.configuration_internvl_chat import InternVLChatConfig
    from MLP_LLM.modeling_internvl_chat import InternVLChatModel

    config = InternVLChatConfig(
        vision_config=dict(
            image_size=448, patch_size=14, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, use_flash_attn=False,
        ),
        llm_config=dict(
            architectures=["Qwen2ForCausalLM"], vocab_size=len(tokenizer),
            hidden_size=64, intermediate_size=128, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        force_image_size=448, downsample_ratio=0.5,
        template="internvl2_5", ps_version="v2",
    )
    model = InternVLChatModel(config).to(DEVICE).eval()
    qformer = InstructBlipQFormerModel(InstructBlipQFormerConfig(
        hidden_size=64, encoder_hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
    )).to(DEVICE).eval()
    return model, qformer


@torch.inference_mode()
def main():
    # Setup: keep downloaded weights and library settings in the project cache.
    (CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "ultralytics")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    init_logger(str(CACHE_DIR) + os.sep)
    torch.manual_seed(42)
    from Concat_Q_Former.qformer_bridge import _extract_vit_tokens

    print("Simple pipeline: YOLO is pretrained; the neural encoder/LLM weights are untrained.")

    # 1. INPUTS: read nine RGB frames; only the last frame goes to the ViT.
    frames = []
    for path in sorted(IMAGES_DIR.glob("*.jpg"))[:9]:
        with Image.open(path) as image:
            frames.append(np.array(image.convert("RGB")))

    # 2. OBJECT TRACKER: all nine frames contribute to the final motion estimates.
    # Object Tracker
    label_ids, direction_ids, numeric_features, object_mask, num_labels = track_objects(frames)

    # Build the small neural modules used in the following stages.
    tokenizer = make_tokenizer()
    model, qformer = make_models(tokenizer)
    # Tracking Encoder
    tracking_encoder = TrajectoryBackbone(num_labels, 6, num_layers=4).to(DEVICE).eval()
    visual_input_projection = nn.Linear(128, 64).to(DEVICE)
    visual_output_projection = nn.Linear(64, 128).to(DEVICE)
    trajectory_projection = TrajectoryConcatHead(128, 128).to(DEVICE).eval()
    queries = nn.Parameter(torch.randn(1, 32, 64, device=DEVICE) * 0.02)

    # 3. VISION ENCODER: resize/normalize the final frame, then extract tokens.
    # ViT + CLS removal + pixel shuffle: [1,3,448,448] -> [1,256,128].
    image = Image.fromarray(frames[-1]).resize((448, 448), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255)
    pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    visual_tokens = _extract_vit_tokens(model, (pixels - mean) / std)

    # 4. Q-FORMER: 32 learned queries attend to 256 visual tokens.
    # This simple version uses image-only queries; the prompt goes to the LLM.
    # The prompt is not gone to this Q-Former yet
    visual_tokens = qformer(
        input_ids=None,
        query_embeds=queries,
        encoder_hidden_states=visual_input_projection(visual_tokens),
        encoder_attention_mask=torch.ones((1, 256), dtype=torch.long, device=DEVICE),
        return_dict=True,
    ).last_hidden_state
    visual_tokens = visual_output_projection(visual_tokens)  # [1,32,128]

    # 5. TRACKING ENCODER: label/direction embeddings + numeric MLP + slot
    # embeddings + four self-attention layers produce six trajectory tokens.
    trajectory_tokens = tracking_encoder(label_ids, direction_ids, numeric_features, object_mask)
    trajectory_tokens = trajectory_projection(trajectory_tokens)
    trajectory_tokens = trajectory_tokens * object_mask.unsqueeze(-1)  # [1,6,128]

    # 6. FUSION: concatenate along the token axis, then apply the shared MLP.
    # [1,32,128] + [1,6,128] -> [1,38,128] -> [1,38,64].
    fused_tokens = torch.cat([visual_tokens, trajectory_tokens], dim=1)
    fused_mask = torch.cat([torch.ones((1, 32), dtype=torch.long, device=DEVICE), object_mask], dim=1)
    projected_tokens = model.mlp1(fused_tokens) * fused_mask.unsqueeze(-1)

    # 7. LLM INPUT: replace 38 placeholder embeddings with the projected tokens.
    prompt = "<IMG_CONTEXT>" * 38 + "\n" + PROMPT
    text = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    embeddings = model.language_model.get_input_embeddings()(text.input_ids)
    image_positions = text.input_ids == tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    embeddings[image_positions] = projected_tokens.reshape(38, 64)
    text.attention_mask[image_positions] = fused_mask.flatten()

    # 8. OUTPUT: generate token IDs and decode them into text.
    output_ids = model.language_model.generate(
        inputs_embeds=embeddings, attention_mask=text.attention_mask,
        max_new_tokens=32, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated text (untrained):", repr(answer))


if __name__ == "__main__":
    main()
