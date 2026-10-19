"""Long-lived, single-stream navigation runtime.

    from main import get_pipeline
    runtime = get_pipeline()                  # loads weights once per process
    runtime.reset_stream("camera-1")
    result = runtime.process_frame(rgb_frame, stream_id="camera-1", generate=True)

Call process_frame serially for every new RGB uint8 frame. Guidance generation
does not reset temporal state. Call reset_stream for a seek, replay or scene cut.
Changing stream_id or source dimensions also resets temporal state, not models.
"""

import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import torch


class StreamingPipeline:
    def __init__(self, backend=None):
        if backend is None:
            import main as backend
        self.backend = backend
        self._lock = threading.RLock()
        (backend.CACHE_DIR / "ultralytics").mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(backend.CACHE_DIR / "ultralytics")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        backend.init_logger(str(backend.CACHE_DIR) + os.sep)
        print("Loading navigation models (once per worker)...", flush=True)
        self.label_vocab, self.direction_vocab = backend.load_vocabularies()
        self.tracking_bundle = backend.load_tracking_encoder(backend.TRACKING_HEADS)
        self.model_bundle = backend.load_models()
        self.detector, self.tracker_path = backend.load_detector()
        self.stc_cache = getattr(self.model_bundle[0].vision_model, "_stc_cacher", None)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=backend.DEVICE).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=backend.DEVICE).view(1, 3, 1, 1)
        self.reset_stream()
        print("Navigation models ready; subsequent requests reuse these instances.", flush=True)

    def reset_stream(self, stream_id=None):
        """Clear temporal state while retaining all model weights, including YOLO."""
        with self._lock:
            if self.stc_cache is not None:
                self.stc_cache.reset()
            predictor = getattr(self.detector, "predictor", None)
            if predictor is not None:
                for tracker in getattr(predictor, "trackers", ()):
                    tracker.reset()
                # persist=True keeps the predictor/model; discard prior results.
                predictor.results = None
            self.stream_id = stream_id
            self.frame_index = 0
            self._shape = None
            self._last_frame_id = None
            self._visual_tokens = None
            self._tracking_inputs = None

    def _encode_frame(self, frame):
        from Concat_Q_Former.qformer_bridge import _extract_vit_tokens

        image = Image.fromarray(frame).resize((448, 448), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0).to(self.backend.DEVICE)
        return _extract_vit_tokens(self.model_bundle[0], ((pixels - self.mean) / self.std).to(self.backend.DTYPE))

    @torch.inference_mode()
    def process_frame(self, frame, *, stream_id="default", frame_id=None, generate=False):
        """Consume one frame; retain only its features and bounded STC/tracker state.

        Optional frame_id must increase within a stream (helps reject retries or
        out-of-order frames). A repeated scene must explicitly start a new stream.
        Calls serialize; callers should apply backpressure instead of queueing video.
        """
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0:
            raise ValueError("frame must be a nonempty H x W x 3 RGB uint8 array")
        if frame_id is not None and (isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0):
            raise ValueError("frame_id must be a nonnegative integer")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a nonempty string")
        with self._lock:
            if self.stream_id != stream_id or (self._shape is not None and self._shape != frame.shape):
                self.reset_stream(stream_id)
            if frame_id is not None and self._last_frame_id is not None and frame_id <= self._last_frame_id:
                raise ValueError("frame_id must increase; reset_stream before replay or seek")
            try:
                self._tracking_inputs = self.backend.track_objects(
                    [frame], self.label_vocab, self.direction_vocab,
                    detector=self.detector, tracker_path=self.tracker_path,
                )
                self._visual_tokens = self._encode_frame(frame)
                self._shape = frame.shape
                self._last_frame_id = frame_id if frame_id is not None else self._last_frame_id
                self.frame_index += 1
                answer = self.generate_guidance() if generate else None
                return {"frame_index": self.frame_index, "answer": answer,
                        "stc_mode": self.stc_cache.last_mode if self.stc_cache is not None else "disabled"}
            except Exception:
                # A failed frame may have advanced one branch but not the other.
                self.reset_stream(stream_id)
                raise

    @torch.inference_mode()
    def generate_guidance(self):
        """Generate from the last frame; leave tracking/STC state available."""
        with self._lock:
            if self._visual_tokens is None:
                raise ValueError("Process a frame before requesting guidance")
            return self.backend.generate_guidance(
                self.model_bundle, self.tracking_bundle, self._visual_tokens, self._tracking_inputs,
            )

    def run_directory(self, images_dir, *, limit=9, on_progress=None):
        """Replay a directory as a NEW video, reusing all loaded models."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        images_dir = Path(images_dir).resolve()
        paths = sorted(images_dir.glob("*.jpg"))
        if limit:
            paths = paths[:limit]
        if not paths:
            raise ValueError(f"No JPG frames found in {images_dir}")
        stream_id = str(images_dir)
        with self._lock:
            self.reset_stream(stream_id)
            try:
                for index, path in enumerate(paths):
                    with Image.open(path) as image:
                        frame = np.array(image.convert("RGB"))
                    result = self.process_frame(frame, stream_id=stream_id, frame_id=index,
                                                generate=index == len(paths) - 1)
                    if on_progress is not None:
                        on_progress(index + 1, len(paths))
                return result["answer"]
            except Exception:
                self.reset_stream(stream_id)
                raise

    def run_video(self, source, *, guidance_every=9, on_result=None):
        """Read a video/camera sequentially, with no unbounded frame queue.

        This loop waits for inference before reading again; it makes no realtime
        latency guarantee. Ctrl+C releases the capture and temporal state.
        """
        import cv2

        if isinstance(guidance_every, bool) or not isinstance(guidance_every, int) or guidance_every < 1:
            raise ValueError("guidance_every must be a positive integer")
        capture = cv2.VideoCapture(source)
        stream_id = f"capture:{source}"

        def emit(result):
            if on_result is not None:
                on_result(result)
            elif result["answer"] is not None:
                print("Generated text:", result["answer"], flush=True)

        with self._lock:
            try:
                self.reset_stream(stream_id)
                if not capture.isOpened():
                    raise ValueError(f"Cannot open video/camera: {source}")
                index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    result = self.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                                stream_id=stream_id, frame_id=index,
                                                generate=(index + 1) % guidance_every == 0)
                    emit(result)
                    index += 1
                if index == 0:
                    raise ValueError(f"No frames received from video/camera: {source}")
                if index % guidance_every:
                    emit({"frame_index": index, "answer": self.generate_guidance()})
            finally:
                capture.release()
                self.reset_stream()
