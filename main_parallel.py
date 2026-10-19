"""Run independent vision and tracking branches concurrently.

    python main_parallel.py
    python main_parallel.py --mode sequential
    python main_parallel.py --runs 2

Weights are loaded once per Python process and reused by subsequent main() calls.
The Hugging Face disk cache is separate from this RAM/VRAM model cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

# Importing the original entry point does not execute its main(). Reuse its
# checkpoint loaders, vocabulary and constants without changing that pipeline.
import main as pipeline
import numpy as np
import torch
import yaml
from PIL import Image

IMAGES_DIR = pipeline.IMAGES_DIR
_runtime = None
_load_lock = Lock()


class FrameTracker:
    """Single-owner YOLO/BoT-SORT runner for the pinned Ultralytics 8.3.203.

    Use the same preprocessing, inference, postprocessing and tracking hooks as
    YOLO.track(). Its standard prediction loop uses device-wide synchronization
    for profiling, which would also wait for the independent vision stream.
    This image-only loop omits profiling and rendering; it does not patch YOLO.
    """

    @torch.inference_mode()
    def __init__(self, sample_frame):
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionPredictor
        from ultralytics.trackers.track import on_predict_start
        from ultralytics.utils import ROOT as YOLO_ROOT

        config = yaml.safe_load((YOLO_ROOT / "cfg/trackers/botsort.yaml").read_text())
        config.update(with_reid=True, model="auto")
        tracker_path = pipeline.CACHE_DIR / "botsort_reid.yaml"
        tracker_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        detector = YOLO(str(pipeline.CACHE_DIR / "yolo11n.pt"))
        self.predictor = DetectionPredictor(overrides={
            **detector.overrides,
            "mode": "track", "conf": 0.1, "batch": 1, "rect": True,
            "device": pipeline.DEVICE, "tracker": str(tracker_path),
            "verbose": False, "save": False,
        })
        predictor = self.predictor
        predictor.setup_model(model=detector.model, verbose=False)
        predictor.setup_source(np.ascontiguousarray(sample_frame[:, :, ::-1]))
        predictor.model.warmup(imgsz=(1, predictor.model.ch, *predictor.imgsz))
        predictor.done_warmup = True
        on_predict_start(predictor, persist=True)

    def reset(self):
        # Each main() invocation is an independent clip. Preserve IDs across
        # its frames, but never carry trajectories over from a previous clip.
        for tracker in self.predictor.trackers:
            tracker.reset()
        self.predictor._feats = None

    def track(self, frame):
        from ultralytics.trackers.track import on_predict_postprocess_end

        predictor = self.predictor
        predictor.setup_source(np.ascontiguousarray(frame[:, :, ::-1]))
        predictor.batch = next(iter(predictor.dataset))
        originals = predictor.batch[1]
        pixels = predictor.preprocess(originals)
        predictions = predictor.inference(pixels)
        predictor.results = predictor.postprocess(predictions, pixels, originals)
        # Includes native detector-feature ReID and the normal BoT-SORT update.
        on_predict_postprocess_end(predictor, persist=True)
        return predictor.results[0]


@dataclass
class Runtime:
    label_vocab: dict
    direction_vocab: dict
    tracking_encoder: object
    trajectory_projection: object
    model: object
    tokenizer: object
    qformer: object
    queries: torch.Tensor
    qtokenizer: object
    projections: object
    tracker: FrameTracker
    visual_stream: object = None
    tracking_stream: object = None
    # The cached tracker and predictor are mutable: only one clip may own them.
    run_lock: object = field(default_factory=Lock)


@torch.inference_mode()
def get_runtime(sample_frame):
    """Reuse resident models; publish the cache only after loading succeeds."""
    global _runtime
    with _load_lock:
        if _runtime is not None:
            print("Models already loaded in this process; reusing RAM/VRAM weights.")
            return _runtime

        print("Models are not loaded in this process; loading once...")
        (pipeline.CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(pipeline.CACHE_DIR / "ultralytics")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        pipeline.init_logger(str(pipeline.CACHE_DIR) + os.sep)
        started = perf_counter()
        label_vocab, direction_vocab = pipeline.load_vocabularies()
        tracking_encoder, trajectory_projection = pipeline.load_tracking_encoder(pipeline.TRACKING_HEADS)
        model, tokenizer, qformer, queries, qtokenizer, projections = pipeline.load_models()
        tracker = FrameTracker(sample_frame)
        visual_stream = tracking_stream = None
        if torch.device(pipeline.DEVICE).type == "cuda":
            visual_stream = torch.cuda.Stream(device=pipeline.DEVICE)
            tracking_stream = torch.cuda.Stream(device=pipeline.DEVICE)
            # Finish initialization before handing model storage to workers.
            torch.cuda.synchronize(pipeline.DEVICE)
        runtime = Runtime(
            label_vocab, direction_vocab, tracking_encoder, trajectory_projection,
            model, tokenizer, qformer, queries, qtokenizer, projections, tracker,
            visual_stream, tracking_stream,
        )
        _runtime = runtime
        print(f"Model initialization: {perf_counter() - started:.3f}s")
        return runtime


def run_visual_branch(runtime, frame):
    """Final RGB frame -> ViT -> prompt-aware Q-Former -> visual MLP."""
    from Concat_Q_Former.qformer_bridge import _extract_vit_tokens

    device, dtype = pipeline.DEVICE, pipeline.DTYPE
    image = Image.fromarray(frame).resize((448, 448), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tokens = _extract_vit_tokens(runtime.model, ((pixels - mean) / std).to(dtype))
    visual_input = runtime.projections["qformer_input_proj"](tokens.float()).to(dtype)
    qtext = runtime.qtokenizer(pipeline.PROMPT, return_tensors="pt", truncation=True, max_length=128).to(device)
    query_mask = torch.ones((1, 32), dtype=torch.long, device=device)
    compressed = runtime.qformer(
        input_ids=qtext.input_ids, query_embeds=runtime.queries,
        attention_mask=torch.cat([query_mask, qtext.attention_mask], dim=1),
        encoder_hidden_states=visual_input,
        encoder_attention_mask=torch.ones((1, 256), dtype=torch.long, device=device),
        return_dict=True,
    ).last_hidden_state[:, :32]
    tokens = runtime.projections["qformer_to_mlp1_proj"](compressed.float())
    return runtime.model.mlp1(tokens.to(dtype))


def run_tracking_branch(runtime, frames):
    """Chronological frames -> YOLO/BoT-SORT -> tracking encoder/projector."""
    runtime.tracker.reset()
    for frame in frames:
        result = runtime.tracker.track(frame)

    device = pipeline.DEVICE
    label_ids = torch.zeros((1, 6), dtype=torch.long, device=device)
    direction_ids = torch.zeros_like(label_ids)
    numeric_features = torch.zeros((1, 6, 6), dtype=torch.float32, device=device)
    object_mask = torch.zeros_like(label_ids)
    height, width = frames[-1].shape[:2]
    boxes = result.boxes
    if boxes is not None and boxes.is_track:
        states = {int(t.track_id): t for t in runtime.tracker.predictor.trackers[0].tracked_stracks}
        for slot, index in enumerate(boxes.xyxy[:, 3].argsort(descending=True)[:6].tolist()):
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            track_id = int(boxes.id[index].item())
            vx, vy = map(float, states[track_id].mean[4:6])
            bearing = math.degrees(math.atan2((x1 + x2) / 2 - width / 2, width / 2))
            clock = pipeline.CLOCK_POSITIONS[sum(bearing > edge for edge in (-30, -10, 10, 30))]
            label = result.names[int(boxes.cls[index].item())]
            label = pipeline.DETECTOR_LABEL_ALIASES.get(label, label)
            label_ids[0, slot] = runtime.label_vocab.get(label, 0)
            direction_ids[0, slot] = runtime.direction_vocab[clock]
            numeric_features[0, slot] = torch.tensor([
                x1 / width, y1 / height, x2 / width, y2 / height,
                math.atan2(vy, vx) / math.pi,
                100 * math.hypot(vx, vy) / math.hypot(width, height),
            ], device=device)
            object_mask[0, slot] = 1
    tokens = runtime.tracking_encoder(label_ids, direction_ids, numeric_features, object_mask)
    return runtime.trajectory_projection(tokens) * object_mask.unsqueeze(-1)


@torch.inference_mode()
def _run_worker(branch, stream, *args):
    # inference_mode is thread-local and must be enabled inside each worker.
    started = perf_counter()
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        try:
            tokens = branch(*args)
        finally:
            # A future represents completed GPU work, including on failure.
            # Waiting on this stream leaves the other branch free to continue.
            if stream is not None:
                stream.synchronize()
    return tokens, perf_counter() - started


def encode_branches(runtime, frames, mode="parallel"):
    """Caller owns runtime.run_lock until fusion/generation have completed."""
    if mode not in {"parallel", "sequential"}:
        raise ValueError("mode must be 'parallel' or 'sequential'")
    if runtime.visual_stream is not None:
        current = torch.cuda.current_stream(pipeline.DEVICE)
        runtime.visual_stream.wait_stream(current)
        runtime.tracking_stream.wait_stream(current)
    started = perf_counter()
    if mode == "parallel":
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="navsvlm") as pool:
            visual = pool.submit(_run_worker, run_visual_branch, runtime.visual_stream, runtime, frames[-1])
            tracking = pool.submit(_run_worker, run_tracking_branch, runtime.tracking_stream, runtime, frames)
            visual_tokens, visual_time = visual.result()
            trajectory_tokens, tracking_time = tracking.result()
    else:
        visual_tokens, visual_time = _run_worker(run_visual_branch, runtime.visual_stream, runtime, frames[-1])
        trajectory_tokens, tracking_time = _run_worker(run_tracking_branch, runtime.tracking_stream, runtime, frames)
    if runtime.visual_stream is not None:
        # Worker synchronization makes the values ready; record_stream also
        # protects their storage while fusion consumes them on the main stream.
        current.wait_stream(runtime.visual_stream)
        current.wait_stream(runtime.tracking_stream)
        visual_tokens.record_stream(current)
        trajectory_tokens.record_stream(current)
    print(f"Branches ({mode}): {perf_counter() - started:.3f}s "
          f"[vision {visual_time:.3f}s, tracking {tracking_time:.3f}s]")
    return visual_tokens, trajectory_tokens


def generate_answer(runtime, visual_tokens, trajectory_tokens):
    """Preserve main.py's token order, padded slots, prompt and decoding."""
    from MLP_LLM.conversation import get_conv_template

    fused_tokens = torch.cat([visual_tokens, trajectory_tokens.to(pipeline.DTYPE)], dim=1)
    template = get_conv_template(runtime.model.template)
    template.system_message = "You are a navigation assistant for visually impaired users."
    template.append_message(template.roles[0], "<img>" + "<IMG_CONTEXT>" * 38 + "</img>\n" + pipeline.PROMPT)
    template.append_message(template.roles[1], None)
    text = runtime.tokenizer(template.get_prompt(), return_tensors="pt").to(pipeline.DEVICE)
    embeddings = runtime.model.language_model.get_input_embeddings()(text.input_ids)
    image_positions = text.input_ids == runtime.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    embeddings[image_positions] = fused_tokens.reshape(38, 2048).to(embeddings)
    stop_id = runtime.tokenizer.convert_tokens_to_ids(template.sep)
    output_ids = runtime.model.language_model.generate(
        inputs_embeds=embeddings, attention_mask=text.attention_mask,
        max_new_tokens=128, do_sample=False, use_cache=True,
        eos_token_id=stop_id, pad_token_id=stop_id,
    )
    return runtime.tokenizer.decode(output_ids[0], skip_special_tokens=True).split(template.sep)[0].strip()


@torch.inference_mode()
def main(images_dir=None, mode="parallel"):
    if mode not in {"parallel", "sequential"}:
        raise ValueError("mode must be 'parallel' or 'sequential'")
    images_dir = Path(images_dir) if images_dir is not None else IMAGES_DIR
    paths = sorted(images_dir.glob("*.jpg"))[:9]
    if not paths:
        raise ValueError(f"No JPG frames found in {images_dir}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.array(image.convert("RGB")))

    runtime = get_runtime(frames[-1])
    with runtime.run_lock:
        started = perf_counter()
        visual_tokens, trajectory_tokens = encode_branches(runtime, frames, mode)
        answer = generate_answer(runtime, visual_tokens, trajectory_tokens)
        print(f"Inference including LLM (excluding model loading): {perf_counter() - started:.3f}s")
        print("Generated text:", answer)
        return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--mode", choices=("parallel", "sequential"), default="parallel")
    parser.add_argument("--runs", type=int, default=1, help="Repeat inference in one process, reusing loaded models.")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")
        main(args.images_dir, args.mode)
