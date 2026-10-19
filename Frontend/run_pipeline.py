"""Persistent JSON-lines worker, or one-shot directory inference.

Worker stdin requests: {id, command: "run", images_dir, limit?},
{id, command: "frame", path, stream_id, frame_id?, generate?},
{id, command: "reset", stream_id?}, or {command: "shutdown"}.
Frame requests retain state across messages. Run requests replay a new sequence.
Only local trusted callers should send paths. Stdout events are prefixed with
NAVSVLM_EVENT; model logs go to stderr. One request is processed at a time.
"""
import argparse
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import traceback

EVENT_PREFIX = "NAVSVLM_EVENT "


def load_pipeline(pipeline_path):
    pipeline_path = Path(pipeline_path).resolve()
    # main.py's imports must resolve as they do when run from the project root.
    sys.path.insert(0, str(pipeline_path.parent))
    spec = importlib.util.spec_from_file_location("navsvlm_demo_pipeline", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    return pipeline


def run_pipeline(pipeline_path, images_dir):
    pipeline = load_pipeline(pipeline_path)
    pipeline.main(images_dir=Path(images_dir).resolve())


def serve(pipeline_path):
    protocol = sys.stdout

    def emit(event, request_id=None, **payload):
        protocol.write(EVENT_PREFIX + json.dumps({"event": event, "id": request_id, **payload}, ensure_ascii=True) + "\n")
        protocol.flush()

    # Import/load once, lazily on the first request. Keep model output separate
    # from completion messages; a multiline answer is a single JSON event.
    pipeline = None
    runtime = None
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"] if "id" in request else None
            command = request.get("command")
            if command == "shutdown":
                break
            if command not in {"run", "frame", "reset"}:
                raise ValueError("Unknown worker command")
            with redirect_stdout(sys.stderr):
                if pipeline is None:
                    pipeline = load_pipeline(pipeline_path)
                if runtime is None:
                    runtime = pipeline.get_pipeline()
                    emit("ready", request_id)
                if command == "run":
                    answer = runtime.run_directory(
                        Path(request["images_dir"]).resolve(), limit=request.get("limit", 9),
                        on_progress=lambda done, total: emit("progress", request_id, frames_done=done, frames_total=total),
                    )
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("The pipeline finished without generated guidance")
                    emit("complete", request_id, answer=answer)
                elif command == "frame":
                    import numpy as np
                    from PIL import Image

                    with Image.open(request["path"]) as image:
                        frame = np.array(image.convert("RGB"))
                    result = runtime.process_frame(
                        frame, stream_id=request["stream_id"], frame_id=request.get("frame_id"),
                        generate=request.get("generate", False),
                    )
                    emit("complete", request_id, **result)
                else:
                    runtime.reset_stream(request.get("stream_id"))
                    emit("complete", request_id)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            # Preserve loaded weights, discard possibly inconsistent stream state.
            if runtime is not None:
                try:
                    runtime.reset_stream()
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    emit("error", request_id, error=f"{exc}; worker reset failed")
                    return 1
            emit("error", request_id, error=str(exc))
    if runtime is not None:
        with redirect_stdout(sys.stderr):
            runtime.reset_stream()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        sys.exit(serve(args.pipeline))
    if args.images_dir is None:
        parser.error("--images-dir is required without --worker")
    run_pipeline(args.pipeline, args.images_dir)
