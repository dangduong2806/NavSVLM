"""Manual benchmark of native InternViT versus STC on real consecutive frames.

Run from the repository root (this script never starts automatically):
    python benchmark_stc.py --images-dir wad_sample/images --profile-layers

Loads one pretrained InternVL base on CPU, keeps its vision tower and pixel
shuffle, and releases the language model/projector before moving vision to the
target device. No YOLO, Q-Former or language generation is run. HF is offline by
default; --online explicitly allows downloads. FP32/eager attention is the
default for the initial correctness check. Results describe vision encoding,
not end-to-end guidance speed or navigation accuracy.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parent
MIB = 1024 ** 2


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure(vision, ratio, interval):
    previous = getattr(vision, "_stc_cacher", None)
    if previous is not None:
        previous.remove()
    if ratio is None:
        return None
    return enable_stc_cacher(vision, update_token_ratio=ratio, cache_interval=interval)


@contextmanager
def native_forward(cache):
    """Temporarily bypass the adapter WITHOUT clearing its temporal reference.

    Used only for correctness comparisons, never for the timed baseline.
    The baseline and selective pass therefore use exactly the same weights.
    """
    if cache._active:
        raise RuntimeError("Cannot switch forwards during an active frame")
    modules = (cache.encoder, *cache.layers)
    saved = [module.forward for module in modules]
    originals = (cache._original_encoder_forward, *cache._original_layer_forwards)
    try:
        for module, forward in zip(modules, originals):
            module.forward = forward
        yield
    finally:
        for module, forward in zip(modules, saved):
            module.forward = forward


def shuffle(model, tokens):
    patches = tokens[:, 1:]
    side = math.isqrt(patches.shape[1])
    if side * side != patches.shape[1]:
        raise ValueError("Benchmark preprocessing expects a square patch grid")
    patches = patches.reshape(patches.shape[0], side, side, -1)
    patches = model.pixel_shuffle(patches, scale_factor=model.downsample_ratio)
    return patches.reshape(patches.shape[0], -1, patches.shape[-1])


def encode(model, pixels, *, capture_layers=False):
    outputs = model.vision_model(
        pixel_values=pixels, output_hidden_states=capture_layers or model.select_layer != -1,
        return_dict=True,
    )
    selected = (outputs.last_hidden_state if model.select_layer == -1
                else outputs.hidden_states[model.select_layer])
    features = shuffle(model, selected)
    if not capture_layers:
        return features
    # Transfers and comparisons are OUTSIDE all performance measurements.
    return {
        "layers": [state.detach().cpu() for state in outputs.hidden_states],
        "vision": outputs.last_hidden_state.detach().cpu(),
        "features": features.detach().cpu(),
    }


def compare(actual, expected, atol, rtol):
    if actual.shape != expected.shape:
        return {"shape_match": False, "passed": False,
                "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    actual, expected = actual.double(), expected.double()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return {"shape_match": True, "finite": False, "passed": False}
    difference = (actual - expected).abs()
    flat_a, flat_b = actual.flatten(), expected.flatten()
    norm_a, norm_b = flat_a.norm().item(), flat_b.norm().item()
    cosine = (float(torch.dot(flat_a, flat_b)) / (norm_a * norm_b)
              if norm_a and norm_b else (1.0 if norm_a == norm_b else 0.0))
    return {
        "shape_match": True, "shape": list(actual.shape), "finite": True,
        "max_abs_error": difference.max().item(), "mean_abs_error": difference.mean().item(),
        "relative_l2": difference.norm().item() / max(norm_b, 1e-30),
        "cosine": max(-1.0, min(1.0, cosine)),
        "passed": bool((difference <= atol + rtol * expected.abs()).all()),
    }


def audit_outputs(model, frames, args, ratio):
    cache = configure(model.vision_model, ratio, args.cache_interval)
    rows = []
    mlp_lengths = []
    # A second check beyond last_mode: selective MLP must really receive K tokens.
    handle = model.vision_model.encoder.layers[0].mlp.register_forward_pre_hook(
        lambda module, inputs: mlp_lengths.append(inputs[0].shape[1])
    )
    try:
        for frame in frames:
            pixels = frame["pixels"].to(args.device)
            with native_forward(cache):
                baseline = encode(model, pixels, capture_layers=True)
            mlp_lengths.clear()
            actual = encode(model, pixels, capture_layers=True)
            count = baseline["layers"][0].shape[1]
            expected_mode = "reference" if len(rows) % args.cache_interval == 0 else "selective"
            expected_k = count if expected_mode == "reference" or ratio == 1 else 1 + max(1, int((count - 1) * ratio))
            if cache.last_mode != expected_mode or mlp_lengths != [expected_k]:
                raise AssertionError(f"Selective path not exercised correctly: {cache.last_mode}, MLP={mlp_lengths}")
            layers = [compare(a, b, args.atol, args.rtol)
                      for a, b in zip(actual["layers"], baseline["layers"])]
            if len(actual["layers"]) != len(baseline["layers"]):
                raise AssertionError("Hidden-state counts differ")
            row = {
                "frame": frame["name"], "mode": cache.last_mode,
                "mlp_tokens_first_layer": mlp_lengths[0], "layers": layers,
                "vision": compare(actual["vision"], baseline["vision"], args.atol, args.rtol),
                "features": compare(actual["features"], baseline["features"], args.atol, args.rtol),
            }
            row["passed"] = all(item["passed"] for item in [*layers, row["vision"], row["features"]])
            rows.append(row)
            print(f"Output audit ratio={ratio:g}: {frame['name']} ({cache.last_mode})", flush=True)
            del actual, baseline, pixels
    finally:
        handle.remove()
        cache.remove()
    return {"ratio": ratio, "passed": all(row["passed"] for row in rows),
            "selective_frames": sum(row["mode"] == "selective" for row in rows), "frames": rows}


def cache_bytes(cache):
    if cache is None:
        return 0
    return sum(tensor.numel() * tensor.element_size()
               for reference in cache._references.values()
               for tensor in (reference.key, reference.value, reference.attention, reference.mlp))


def warmup(model, frames, cache, args):
    for index in range(args.warmup):
        pixels = frames[index % len(frames)]["pixels"].to(args.device)
        output = encode(model, pixels)
        del output, pixels
    synchronize(args.device)
    if cache is not None:
        cache.reset()


def time_sequence(model, frames, args, ratio):
    cache = configure(model.vision_model, ratio, args.cache_interval)
    try:
        warmup(model, frames, cache, args)
        gc.collect()
        synchronize(args.device)
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        rows = []
        for frame in frames:
            # Decode, normalization and host-to-device transfer are excluded.
            pixels = frame["pixels"].to(args.device)
            synchronize(args.device)
            start = time.perf_counter()
            features = encode(model, pixels)
            synchronize(args.device)
            elapsed = time.perf_counter() - start
            rows.append({"frame": frame["name"], "seconds": elapsed,
                         "mode": cache.last_mode if cache else "baseline"})
            del features, pixels
        return {
            "frames": rows, "total_seconds": sum(row["seconds"] for row in rows),
            "cache_mib": cache_bytes(cache) / MIB,
            "peak_cuda_allocated_mib": (torch.cuda.max_memory_allocated(args.device) / MIB
                                         if args.device.type == "cuda" else None),
        }
    finally:
        if cache is not None:
            cache.remove()


def profile_layers(model, frames, args, ratio):
    """Separate intrusive diagnostic pass, excluded from headline speedup.

    Outer layer timing includes all its attention/projection/selection/merge/
    norm/residual work plus MLP. MLP time is measured by nested hooks. The
    difference is deliberately NOT called pure attention time.
    """
    cache = configure(model.vision_model, ratio, args.cache_interval)
    handles, starts, records = [], {}, {}

    def before(name, count_tokens=False):
        def hook(module, inputs):
            synchronize(args.device)
            if count_tokens:
                records[name]["tokens"] += inputs[0].shape[0] * inputs[0].shape[1]
            starts[name] = time.perf_counter()
        return hook

    def after(name):
        def hook(module, inputs, output):
            synchronize(args.device)
            records[name]["seconds"] += time.perf_counter() - starts[name]
        return hook

    try:
        warmup(model, frames, cache, args)
        for index, layer in enumerate(model.vision_model.encoder.layers):
            for suffix, module in (("layer", layer), ("mlp", layer.mlp)):
                name = f"{index}.{suffix}"
                records[name] = {"seconds": 0.0, "tokens": 0}
                handles.extend((module.register_forward_pre_hook(before(name, suffix == "mlp")),
                                module.register_forward_hook(after(name))))
        for frame in frames:
            pixels = frame["pixels"].to(args.device)
            output = encode(model, pixels)
            del output, pixels
        layer_s = sum(row["seconds"] for name, row in records.items() if name.endswith(".layer"))
        mlp_s = sum(row["seconds"] for name, row in records.items() if name.endswith(".mlp"))
        return {"layer_seconds": layer_s, "mlp_seconds": mlp_s,
                "other_layer_seconds": layer_s - mlp_s,
                "mlp_token_layer_evaluations": sum(row["tokens"] for row in records.values()),
                "per_layer": records,
                "note": "Intrusive synchronized hooks; not headline timing. Other includes attention, projections, selection, cache copies, norms, residuals and hook overhead."}
    finally:
        for handle in handles:
            handle.remove()
        if cache is not None:
            cache.remove()


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def summarize(runs):
    summary = {}
    baseline_total = sum(run["total_seconds"] for run in runs["baseline"])
    for mode, trials in runs.items():
        rows = [row for trial in trials for row in trial["frames"]]
        values = [row["seconds"] for row in rows]
        total = sum(values)
        peaks = [trial["peak_cuda_allocated_mib"] for trial in trials if trial["peak_cuda_allocated_mib"] is not None]
        summary[mode] = {
            "mean_sequence_seconds": statistics.mean(trial["total_seconds"] for trial in trials),
            "mean_frame_ms": statistics.mean(values) * 1000,
            "median_frame_ms": statistics.median(values) * 1000,
            "p95_frame_ms": percentile(values, 0.95) * 1000,
            "encoding_fps": len(values) / total, "speedup": baseline_total / total,
            "latency_reduction_percent": (1 - total / baseline_total) * 100,
            "cache_mib": max(trial["cache_mib"] for trial in trials),
            "peak_cuda_allocated_mib": max(peaks) if peaks else None,
        }
        for kind in ("reference", "selective"):
            subset = [row["seconds"] for row in rows if row["mode"] == kind]
            summary[mode][f"{kind}_mean_ms"] = statistics.mean(subset) * 1000 if subset else None
    return summary


def render_report(report):
    lines = ["# InternViT / STC comparison", "", f"Status: `{report['status']}`.", "",
             "Measured region: InternViT + pixel shuffle, with prepared single-tile 448px inputs. Includes reference refresh and cache overhead; excludes loading, preprocessing, transfers, YOLO, Q-Former and LLM.", ""]
    if "error" in report:
        lines += [f"Error: {report['error']}", ""]
    if "validation" in report:
        validation = report["validation"]
        lines += [f"Selective 100% validation: **{'PASS' if validation['passed'] else 'FAIL'}**, {validation['selective_frames']} selective frames. Tolerances and per-layer errors are in JSON.", ""]
    if "summary" in report:
        lines += ["| Mode | Mean sequence (s) | Mean frame (ms) | Speedup | Cache (MiB) |",
                  "|---|---:|---:|---:|---:|"]
        for mode, row in report["summary"].items():
            lines.append(f"| {mode} | {row['mean_sequence_seconds']:.4f} | {row['mean_frame_ms']:.2f} | {row['speedup']:.3f}x | {row['cache_mib']:.1f} |")
        lines += ["", "Speedup > 1 means faster vision encoding; < 1 means slower. This does not measure end-to-end guidance latency.", ""]
        lines += ["| Mode | Reference mean (ms) | Selective mean (ms) | Peak CUDA allocated (MiB) |",
                  "|---|---:|---:|---:|"]
        for mode, row in report["summary"].items():
            values = ["-" if row[key] is None else f"{row[key]:.2f}" for key in
                      ("reference_mean_ms", "selective_mean_ms", "peak_cuda_allocated_mib")]
            lines.append(f"| {mode} | " + " | ".join(values) + " |")
        lines += ["", "Cache is persistent reference storage only. CUDA peak includes resident vision weights, inputs, outputs, cache and temporary tensors; CPU peak memory is not measured.", ""]
    if "quality" in report:
        rows = [row["features"] for row in report["quality"]["frames"] if row["mode"] == "selective"]
        valid = [row for row in rows if row.get("finite") and row.get("shape_match")]
        if valid:
            lines += [f"STC selective-frame feature cosine: mean {statistics.mean(row['cosine'] for row in valid):.6f}; minimum {min(row['cosine'] for row in valid):.6f}.", ""]
        lines += ["Feature similarity is an agreement check against baseline, not a navigation-accuracy score.", ""]
    if "profiles" in report:
        lines += ["## Separate layer diagnostic (instrumented)", "",
                  "| Mode | MLP (s) | MLP speedup | Other layer work (s) | MLP token-layer evaluations |",
                  "|---|---:|---:|---:|---:|"]
        for mode, row in report["profiles"].items():
            base = report["profiles"].get("baseline")
            speedup = base["mlp_seconds"] / row["mlp_seconds"] if base and row["mlp_seconds"] > 0 else None
            speedup_text = f"{speedup:.3f}x" if speedup is not None else "-"
            lines.append(f"| {mode} | {row['mlp_seconds']:.4f} | {speedup_text} | {row['other_layer_seconds']:.4f} | {row['mlp_token_layer_evaluations']} |")
        lines += ["", "Other layer work includes attention, projections, selection, cache copying, normalization, residuals and hook overhead. These diagnostic timings are not used for headline speedup.", ""]
    return "\n".join(lines) + "\n"


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".md").write_text(render_report(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "wad_sample/images")
    parser.add_argument("--limit", type=int, default=9, help="First N sorted JPGs; 0 uses all")
    parser.add_argument("--model", default="OpenGVLab/InternVL2_5-2B")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--cache-interval", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--profile-layers", action="store_true")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stc_benchmark.json")
    args = parser.parse_args()
    if args.limit < 0 or args.cache_interval < 2 or args.repeats < 1 or args.warmup < 2 or args.threads < 1:
        parser.error("Require limit >= 0, cache-interval >= 2, repeats >= 1, warmup >= 2, threads >= 1")
    if not 0 < args.ratio < 1 or not all(math.isfinite(v) and v >= 0 for v in (args.atol, args.rtol)):
        parser.error("Require 0 < ratio < 1 and finite nonnegative tolerances")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json (a companion .md is also written)")
    paths = sorted(args.images_dir.glob("*.jpg"))
    paths = paths[:args.limit] if args.limit else paths
    if len(paths) < 2:
        parser.error("Need at least two consecutive JPG frames to exercise selective recomputation")

    os.environ.update(USE_TF="0", USE_TORCH="1",
                      HF_HUB_OFFLINE="0" if args.online else "1",
                      TRANSFORMERS_OFFLINE="0" if args.online else "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".model_cache/modules"))
    global torch, enable_stc_cacher
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoModel
    from logutil import init_logger
    from Vision_Encoder.stc_cacher import enable_stc_cacher

    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.device = torch.device(args.device)
    if args.device.type not in {"cpu", "cuda"}:
        parser.error("This benchmark supports cpu or cuda devices")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    init_logger(str(args.output.parent) + os.sep)
    report = {"status": "starting", "settings": settings, "runs": {},
              "notes": ["Same weights, inputs, eager attention and preprocessing for every mode.",
                        "Warmups excluded; cache reset before each measured sequence; all reference frames included.",
                        "Output validation and layer diagnostics run outside performance measurement.",
                        "No tracking-guided selection, text quality scoring or end-to-end timing."]}
    save_report(args.output, report)
    model = None
    try:
        dtype = getattr(torch, args.dtype)
        print("Loading pretrained base once on CPU (excluded from timing)...", flush=True)
        model = AutoModel.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True,
            use_flash_attn=False, cache_dir=str(ROOT / ".model_cache"), local_files_only=not args.online,
        ).eval()
        # Keep the original model's exact pixel_shuffle and select_layer semantics.
        del model.language_model
        del model.mlp1
        gc.collect()
        model.vision_model.to(device=args.device, dtype=dtype).eval()
        if any(layer.attn.use_flash_attn for layer in model.vision_model.encoder.layers):
            raise RuntimeError("Expected eager attention in both baseline and selective paths")
        torch.set_num_threads(args.threads)
        source = Path(inspect.getfile(type(model.vision_model)))
        report["environment"] = {
            "torch": str(torch.__version__), "platform": platform.platform(),
            "device": str(args.device), "dtype": args.dtype, "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
            "attention": "eager", "tf32": False,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "vision_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256((ROOT / "Vision_Encoder/stc_cacher.py").read_bytes()).hexdigest(),
        }
        frames = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
                pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255)
            pixels = pixels.permute(2, 0, 1).unsqueeze(0)
            frames.append({"name": path.name, "pixels": ((pixels - mean) / std).to(dtype)})
        report["inputs"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
        with torch.inference_mode():
            report["status"] = "validating"
            report["validation"] = audit_outputs(model, frames, args, 1.0)
            save_report(args.output, report)
            if not report["validation"]["passed"] or not report["validation"]["selective_frames"]:
                report["status"] = "validation_failed"
                report["error"] = "Selective 100% differs beyond tolerances. Inspect per-layer errors; do not treat adapter correctness as confirmed."
                save_report(args.output, report)
                print(render_report(report))
                return 1
            report["quality"] = audit_outputs(model, frames, args, args.ratio)
            if any(not row["features"].get("finite") or not row["features"].get("shape_match")
                   for row in report["quality"]["frames"]):
                raise RuntimeError("STC produced invalid feature tensors")
            modes = [("baseline", None), ("selective_100", 1.0), ("stc", args.ratio)]
            report["runs"] = {name: [] for name, _ in modes}
            report["status"] = "measuring"
            for repeat in range(args.repeats):
                order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
                for name, ratio in order:
                    print(f"Timing repeat {repeat + 1}/{args.repeats}: {name}", flush=True)
                    result = time_sequence(model, frames, args, ratio)
                    result["repeat"] = repeat + 1
                    result["order_in_repeat"] = [item[0] for item in order]
                    report["runs"][name].append(result)
                    save_report(args.output, report)
            report["summary"] = summarize(report["runs"])
            if args.profile_layers:
                report["profiles"] = {}
                for name, ratio in modes:
                    print(f"Separate layer diagnostic: {name}", flush=True)
                    report["profiles"][name] = profile_layers(model, frames, args, ratio)
                    save_report(args.output, report)
        report["status"] = "complete"
        save_report(args.output, report)
        print(render_report(report))
        print(f"Detailed results: {args.output.resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save_report(args.output, report)
        raise
    finally:
        if model is not None:
            cache = getattr(model.vision_model, "_stc_cacher", None)
            if cache is not None:
                cache.remove()


if __name__ == "__main__":
    raise SystemExit(main())
