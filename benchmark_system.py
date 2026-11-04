"""Đánh giá STC-Cacher và hiệu năng toàn bộ backend NavSVLM.

    python benchmark_system.py --images-dir wad_sample/images --profile-stages

Hai chế độ dùng cùng trọng số, chuỗi ảnh và lịch sinh hướng dẫn. Mặc định sinh
hướng dẫn mỗi 9 khung hình và ở khung cuối. Dùng --guidance-every 4 để đánh giá
cả câu trả lời trên khung tính chọn lọc trong mẫu 9 ảnh, chu kỳ cache 4.

Phần STC dùng lượt đo riêng từng công đoạn (--profile-stages), phân tích thời
gian InternViT + pixel shuffle theo khung tham chiếu/tính chọn lọc và bộ nhớ cache.
Phần toàn hệ thống đo từ đọc ảnh đến văn bản đầu ra, gồm tiền xử lý và truyền
dữ liệu; loại trừ nạp mô hình, làm nóng, đặt lại trạng thái và ghi báo cáo.
Không đo HTTP, giao diện hoặc phát âm thanh. Không tự chạy khi import.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

ROOT = Path(__file__).resolve().parent
MIB = 1024 ** 2
STAGES = {
    "image_decode": "Đọc JPG + giải mã RGB",
    "yolo_tracking": "Object Tracker: YOLO + BoT-SORT/ReID + tạo đặc trưng",
    "vision_preprocess": "Đổi kích thước + chuẩn hóa + truyền dữ liệu",
    "vision_encode": "Vision Encoder: InternViT + pixel shuffle",
    "qformer_visual_projection": "Q-Former + phép chiếu thị giác + tách token",
    "tracking_encoder": "Tracking Encoder + phép chiếu",
    "fusion_prompt": "Dung hợp + tạo prompt và embedding cho LLM",
    "llm_generate": "LLM: xử lý đầu vào + sinh token",
    "text_decode": "Chuyển token đầu ra thành văn bản",
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_mode(runtime, mode, args):
    """Switch only the vision adapter; retain all loaded model instances."""
    runtime.stage_timings = None
    runtime.reset_stream()
    vision = runtime.model_bundle[0].vision_model
    cache = getattr(vision, "_stc_cacher", None)
    if cache is not None:
        cache.remove()
    runtime.stc_cache = None
    if mode == "stc":
        runtime.stc_cache = enable_stc_cacher(vision, args.ratio, args.cache_interval)
    elif mode != "baseline":
        raise ValueError(f"Unknown mode: {mode}")


def cache_mib(runtime):
    if runtime.stc_cache is None:
        return 0.0
    return sum(t.numel() * t.element_size() for r in runtime.stc_cache._references.values()
               for t in (r.key, r.value, r.attention, r.mlp)) / MIB


def replay(runtime, paths, args, mode, *, diagnostic=False, warmup=False):
    set_mode(runtime, mode, args)
    runtime.reset_stream("system-benchmark")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    recorder = StageTimings(args.device) if diagnostic else None
    runtime.stage_timings = recorder
    rows = []
    synchronize(args.device)
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    try:
        for index, path in enumerate(paths):
            if recorder is not None:
                recorder.events.clear()
            synchronize(args.device)
            frame_start = time.perf_counter()
            with stage_scope(recorder, "image_decode"):
                with Image.open(path) as image:
                    frame = np.array(image.convert("RGB"))
            generate = (index + 1) % args.guidance_every == 0 or index == len(paths) - 1
            result = runtime.process_frame(
                frame, stream_id="system-benchmark", frame_id=index, generate=generate,
            )
            synchronize(args.device)
            frame_end = time.perf_counter()
            frame_seconds = frame_end - frame_start
            effective_threads = torch.get_num_threads()
            if effective_threads != args.threads:
                if not warmup:
                    raise RuntimeError(f"PyTorch threads changed to {effective_threads} during timing; requested {args.threads}")
                # Ultralytics may change CPU threads during lazy predictor setup.
                torch.set_num_threads(args.threads)
            if generate and (not isinstance(result["answer"], str) or not result["answer"].strip()):
                raise RuntimeError(f"No guidance returned for {path.name}")
            rows.append({
                "frame": path.name, "input_index": index + 1,
                "stream_frame_index": result["frame_index"], "seconds": frame_seconds,
                "elapsed_since_sequence_start_seconds": frame_end - start,
                "vision_mode": "baseline" if mode == "baseline" else result["stc_mode"],
                "generated": generate, "answer": result["answer"],
                "generation": result.get("generation"), "torch_threads": effective_threads,
                "stages": list(recorder.events) if recorder is not None else None,
            })
            del frame
        synchronize(args.device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(args.device) / MIB if args.device.type == "cuda" else None
        return {
            "mode": mode, "diagnostic": diagnostic, "warmup": warmup,
            "total_seconds": elapsed, "frames": rows,
            "first_complete_guidance_seconds": next(row["elapsed_since_sequence_start_seconds"] for row in rows if row["generated"]),
            "frame_count": len(rows), "guidance_count": sum(row["generated"] for row in rows),
            "vision_mode_counts": dict(Counter(row["vision_mode"] for row in rows)),
            "cache_end_mib": cache_mib(runtime), "peak_cuda_allocated_mib": peak,
        }
    finally:
        runtime.stage_timings = None


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def summarize(runs):
    baseline_total = sum(run["total_seconds"] for run in runs["baseline"])
    summaries = {}
    for mode, trials in runs.items():
        totals = [run["total_seconds"] for run in trials]
        rows = [row for run in trials for row in run["frames"]]
        frame_times = [row["seconds"] for row in rows]
        guidance_rows = [row for row in rows if row["generated"]]
        ordinary_rows = [row for row in rows if not row["generated"]]
        peaks = [run["peak_cuda_allocated_mib"] for run in trials if run["peak_cuda_allocated_mib"] is not None]
        summaries[mode] = {
            "mean_sequence_seconds": statistics.mean(totals),
            "mean_first_complete_guidance_seconds": statistics.mean(run["first_complete_guidance_seconds"] for run in trials),
            "min_sequence_seconds": min(totals), "max_sequence_seconds": max(totals),
            "stdev_sequence_seconds": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "speedup": baseline_total / sum(totals),
            "latency_reduction_percent": (1 - sum(totals) / baseline_total) * 100,
            "processed_frames_per_second": len(rows) / sum(totals),
            "mean_frame_ms": statistics.mean(frame_times) * 1000,
            "p95_frame_ms": percentile(frame_times, 0.95) * 1000,
            "mean_guidance_frame_ms": statistics.mean(row["seconds"] for row in guidance_rows) * 1000,
            "mean_non_guidance_frame_ms": (statistics.mean(row["seconds"] for row in ordinary_rows) * 1000
                                            if ordinary_rows else None),
            "mean_guidance_count": statistics.mean(run["guidance_count"] for run in trials),
            "returned_output_tokens": sum(row["generation"]["returned_output_tokens"] for row in guidance_rows),
            "mean_returned_tokens_per_answer": statistics.mean(row["generation"]["returned_output_tokens"] for row in guidance_rows),
            "guidance_on_selective_frames": sum(row["vision_mode"] == "selective" for row in guidance_rows),
            "selective_frames": sum(row["vision_mode"] == "selective" for row in rows),
            "cache_end_mib": max(run["cache_end_mib"] for run in trials),
            "peak_cuda_allocated_mib": max(peaks) if peaks else None,
        }
    return summaries


def compare_answers(runs):
    comparisons = []
    baseline = {run["repeat"]: run for run in runs["baseline"]}
    for trial in runs["stc"]:
        base_rows = {row["input_index"]: row for row in baseline[trial["repeat"]]["frames"] if row["generated"]}
        for row in trial["frames"]:
            if not row["generated"]:
                continue
            original = base_rows[row["input_index"]]
            comparisons.append({
                "repeat": trial["repeat"], "frame": row["frame"], "stc_vision_mode": row["vision_mode"],
                "baseline_answer": original["answer"], "stc_answer": row["answer"],
                "exact_match": original["answer"] == row["answer"],
                "baseline_returned_tokens": original["generation"]["returned_output_tokens"],
                "stc_returned_tokens": row["generation"]["returned_output_tokens"],
            })
    return comparisons


def summarize_stages(run):
    stages = {name: {"seconds": 0.0, "calls": 0} for name in STAGES}
    for row in run["frames"]:
        for event in row["stages"]:
            stages[event["stage"]]["seconds"] += event["seconds"]
            stages[event["stage"]]["calls"] += 1
    return stages


def vision_diagnostics(report):
    """Match vision timings by input index, grouped by STC frame mode."""
    profiles = report.get("profiles", {})
    if not all(mode in profiles for mode in ("baseline", "stc")):
        return None

    def seconds(frame):
        values = [event["seconds"] for event in (frame.get("stages") or [])
                  if event["stage"] == "vision_encode"]
        return sum(values) if values else None

    baseline = {frame["input_index"]: seconds(frame)
                for frame in profiles["baseline"]["frames"]}
    groups = {name: [] for name in ("reference", "selective")}
    for frame in profiles["stc"]["frames"]:
        elapsed = seconds(frame)
        base = baseline.get(frame["input_index"])
        if elapsed is not None and base is not None and frame["vision_mode"] in groups:
            groups[frame["vision_mode"]].append((base, elapsed))
    return groups


def markdown(report):
    statuses = {"loading": "Đang nạp mô hình", "warming": "Đang làm nóng",
                "measuring": "Đang đo", "profiling_stages": "Đang đo từng công đoạn",
                "complete": "Hoàn tất", "interrupted": "Đã gián đoạn", "error": "Có lỗi"}
    modes = {"baseline": "Không STC", "stc": "Có STC"}
    settings = report["settings"]
    lines = ["# Đánh giá hiệu năng NavSVLM", "",
             f"Trạng thái: **{statuses.get(report['status'], report['status'])}**.", "",
             "Hai chế độ dùng cùng mô hình, trọng số, thứ tự ảnh và lịch sinh hướng dẫn. Chế độ cơ sở (baseline) tắt STC; cả hai vẫn chạy tracking và mã hóa thị giác trên mọi khung hình.", "",
             f"Thiết bị: `{settings['device']}`; kiểu dữ liệu: `{settings['dtype']}`; số khung hình: {len(report.get('inputs', []))}; số lượt lặp đo toàn hệ thống: {settings['repeats']}. Sinh hướng dẫn mỗi {settings['guidance_every']} khung hình và ở khung cuối.", "",
             f"STC cập nhật {settings['ratio'] * 100:g}% patch cùng token CLS; chu kỳ tham chiếu: {settings['cache_interval']} khung hình.", ""]
    if "error" in report:
        lines += [f"Chi tiết lỗi: {report['error']}", ""]
    if "model_load_seconds" in report:
        lines += [f"Nạp mô hình và khởi tạo runtime một lần: **{report['model_load_seconds']:.2f} giây**. Thời gian này được loại khỏi kết quả bên dưới. Khởi tạo YOLO trì hoãn và lần chạy đầu nằm trong lượt làm nóng, cũng bị loại khỏi phép đo.", ""]

    lines += ["## 1. Hiệu năng STC-Cacher", "",
              "So sánh InternViT cơ sở và InternViT có STC, cùng tính cả pixel shuffle. Số liệu lấy từ lượt đo riêng từng công đoạn của mô hình trong NavSVLM, không gồm tiền xử lý ảnh, truyền dữ liệu, tracker, Q-Former hoặc LLM. Lượt này đồng bộ thiết bị tại ranh giới công đoạn và chạy một lần cho mỗi chế độ, tách khỏi các lượt đo tổng độ trễ ở phần 2.", ""]
    groups = vision_diagnostics(report)
    if groups is not None and "stage_summary" in report:
        lines += ["### Thời gian mã hóa thị giác", "",
                  "| Chế độ | Số lần gọi | Tổng thời gian (s) | Trung bình (ms/lần) | Hệ số tăng tốc |",
                  "|---|---:|---:|---:|---:|"]
        base = report["stage_summary"]["baseline"]["vision_encode"]["seconds"]
        for mode in ("baseline", "stc"):
            row = report["stage_summary"][mode]["vision_encode"]
            mean = f"{row['seconds'] * 1000 / row['calls']:.2f}" if row["calls"] else "—"
            speedup = f"{base / row['seconds']:.3f}×" if row["seconds"] > 0 else "—"
            label = "InternViT cơ sở" if mode == "baseline" else "InternViT + STC"
            lines.append(f"| {label} | {row['calls']} | {row['seconds']:.4f} | {mean} | {speedup} |")
        lines += ["", "### Khung tham chiếu và khung tính chọn lọc", "",
                  "Khung tham chiếu (reference) tính toàn bộ token và lưu cache. Khung tính chọn lọc (selective) chọn token thay đổi để tính lại và tái sử dụng kết quả còn lại. Cache cũng được làm mới khi trạng thái không còn hợp lệ.", "",
                  "| Loại khung STC | Số khung | Baseline cùng vị trí (ms/khung) | STC (ms/khung) | Hệ số tăng tốc |",
                  "|---|---:|---:|---:|---:|"]
        for kind, label in (("reference", "Tham chiếu"), ("selective", "Tính chọn lọc")):
            pairs = groups[kind]
            if not pairs:
                lines.append(f"| {label} | 0 | — | — | — |")
                continue
            base_mean = statistics.mean(pair[0] for pair in pairs)
            stc_mean = statistics.mean(pair[1] for pair in pairs)
            speedup = f"{base_mean / stc_mean:.3f}×" if stc_mean > 0 else "—"
            lines.append(f"| {label} | {len(pairs)} | {base_mean * 1000:.2f} | {stc_mean * 1000:.2f} | {speedup} |")
        lines += ["", "Baseline được ghép với STC theo vị trí ảnh đầu vào; baseline không dùng cache. Hệ số tăng tốc = thời gian baseline / thời gian STC; lớn hơn 1 nghĩa là nhanh hơn.", ""]
        if not groups["selective"]:
            lines += ["**Chưa quan sát được khung tính chọn lọc trong lượt đo công đoạn.** Cần kiểm tra thay đổi kích thước ảnh và việc đặt lại cache.", ""]
    else:
        lines += ["Chưa có đủ số liệu công đoạn. Chạy với `--profile-stages` và chờ hoàn tất để có so sánh InternViT và phân tích khung tham chiếu/tính chọn lọc.", ""]
    lines += ["### Chi phí bộ nhớ đệm (cache overhead)", "",
              "STC lưu K, V, đầu ra attention và MLP theo từng lớp. Chi phí chọn token, lưu/lấy/ghép cache và làm mới reference đã nằm trong thời gian Vision Encoder ở trên; không cộng thêm chúng như một module độc lập.", ""]
    if "summary" in report:
        lines += ["| Chế độ | Bộ nhớ cache STC lớn nhất tại cuối các lượt đo (MiB) |", "|---|---:|"]
        for mode, row in report["summary"].items():
            lines.append(f"| {modes[mode]} | {row['cache_end_mib']:.2f} |")
        lines.append("")
    lines += ["**Chưa đo riêng thời gian thao tác cache.** Chênh lệch giữa baseline và STC là hiệu quả tổng hợp của giảm tính toán, thao tác cache và biến động thực thi; không thể dùng trực tiếp làm số đo thời gian cache. Bộ nhớ cache chỉ tính các tensor tham chiếu được giữ lại, không phải toàn bộ bộ nhớ hệ thống.", "",
              "Để kiểm tra độ đúng ở tỷ lệ cập nhật 100%, sai lệch đặc trưng và thời gian từng lớp trong phép đo riêng nhánh thị giác, dùng `benchmark_stc.py --profile-layers`. Kết quả của phép đo đó có phạm vi riêng và không được cộng vào báo cáo này.", "",
              "## 2. Hiệu năng toàn hệ thống", "",
              "Phạm vi đo: đọc/giải mã JPG → YOLO và tracking → tiền xử lý và mã hóa ảnh → Q-Former → Tracking Encoder → dung hợp → LLM → văn bản đầu ra. Có tính truyền dữ liệu CPU/GPU; loại trừ nạp mô hình, làm nóng, đặt lại trạng thái và ghi báo cáo. Chưa bao gồm HTTP, giao diện trình duyệt hoặc phát âm thanh.", ""]
    if "summary" in report:
        lines += ["### Tổng độ trễ và tốc độ xử lý", "",
                  "| Chế độ | Trung bình cả chuỗi (s) | Nhỏ nhất / lớn nhất (s) | Hệ số tăng tốc | FPS | Token trung bình/câu |",
                  "|---|---:|---:|---:|---:|---:|"]
        for mode, row in report["summary"].items():
            lines.append(f"| {modes[mode]} | {row['mean_sequence_seconds']:.3f} | {row['min_sequence_seconds']:.3f} / {row['max_sequence_seconds']:.3f} | {row['speedup']:.3f}× | {row['processed_frames_per_second']:.3f} | {row['mean_returned_tokens_per_answer']:.1f} |")
        reduction = report["summary"]["stc"]["latency_reduction_percent"]
        lines += ["", f"Mức giảm tổng độ trễ: **{reduction:.2f}%**; giá trị âm nghĩa là chậm hơn. FPS = tổng số khung hình / tổng thời gian của các lượt đo, đã tính việc sinh hướng dẫn theo lịch cấu hình. Đây không phải cam kết xử lý camera theo thời gian thực.", "",
                  "### Độ trễ theo khung hình và bộ nhớ", "",
                  "| Chế độ | Khung sinh hướng dẫn (ms) | Khung còn lại (ms) | Cache STC cuối lượt (MiB) | Bộ nhớ CUDA cấp phát đỉnh (MiB) |",
                  "|---|---:|---:|---:|---:|"]
        for mode, row in report["summary"].items():
            values = ["—" if row[name] is None else f"{row[name]:.2f}" for name in
                      ("mean_guidance_frame_ms", "mean_non_guidance_frame_ms", "cache_end_mib", "peak_cuda_allocated_mib")]
            lines.append(f"| {modes[mode]} | " + " | ".join(values) + " |")
        lines += ["", "Độ trễ khung là trung bình; bộ nhớ lấy giá trị lớn nhất qua các lượt đo. Bộ nhớ CUDA cấp phát đỉnh do PyTorch ghi nhận gồm mô hình đang nạp, cache STC/LLM và tensor tạm, không phải toàn bộ VRAM của tiến trình. **Chưa đo bộ nhớ RAM đỉnh trên CPU**; dấu — là không có số đo, không phải 0.", "",
                  "Thời gian từ đầu chuỗi đến hướng dẫn hoàn chỉnh đầu tiên, không tính nạp mô hình: "
                  + "; ".join(f"{modes[mode]}: {row['mean_first_complete_guidance_seconds']:.3f} s" for mode, row in report["summary"].items())
                  + ". Đây là thời gian chờ câu trả lời hoàn chỉnh, chưa phải thời gian đến token đầu tiên.", ""]
        if report["summary"]["stc"]["selective_frames"] == 0:
            lines += ["**Không có khung tính chọn lọc trong các lượt đo tổng.** Kiểm tra kích thước ảnh và việc đặt lại cache trước khi quy chênh lệch tốc độ cho cơ chế tính chọn lọc.", ""]
        if report["summary"]["stc"]["guidance_on_selective_frames"] == 0:
            lines += ["**Các câu trả lời STC được đo đều sinh trên khung tham chiếu.** Chưa kiểm tra chất lượng hướng dẫn trên khung tính chọn lọc. Với mẫu 9 ảnh, chu kỳ cache 4, dùng `--guidance-every 4` để sinh thêm ở khung 4 và 8; áp dụng cùng lịch cho cả hai chế độ.", ""]
    else:
        lines += ["Chưa hoàn tất các lượt đo tổng độ trễ, FPS và bộ nhớ.", ""]
    if "stage_summary" in report:
        lines += ["### Thời gian từng module và công đoạn phụ", "",
                  "Đây là lượt đo công đoạn riêng có đồng bộ thiết bị, cùng nguồn với phần 1. Không cộng bảng này với phần 1 hoặc dùng tổng của nó thay cho độ trễ toàn hệ thống ở trên.", "",
                  "| Module / công đoạn | Tổng baseline (s) | Tổng STC (s) | Số lần gọi baseline / STC | Baseline (ms/lần) | STC (ms/lần) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, label in STAGES.items():
            base = report["stage_summary"]["baseline"][name]
            stc = report["stage_summary"]["stc"][name]
            means = [f"{row['seconds'] * 1000 / row['calls']:.2f}" if row["calls"] else "—" for row in (base, stc)]
            lines.append(f"| {label} | {base['seconds']:.4f} | {stc['seconds']:.4f} | {base['calls']} / {stc['calls']} | {means[0]} | {means[1]} |")
        profile_tokens = {mode: sum(row["generation"]["returned_output_tokens"] for row in run["frames"] if row["generated"])
                          for mode, run in report["profiles"].items()}
        lines += ["", "Vision Encoder và Object Tracker chạy mỗi khung hình; Q-Former, Tracking Encoder và LLM chạy khi sinh hướng dẫn. Trung bình mỗi lần gọi = tổng thời gian / số lần gọi.", "",
                  "Tổng token đầu ra trong lượt đo công đoạn: " + "; ".join(f"{modes[mode]}: {count}" for mode, count in profile_tokens.items()) + ".", "",
                  "STC trực tiếp thay đổi nhánh mã hóa thị giác. Chênh lệch ở các module khác có thể do biến động thời gian, khác biệt đầu ra số học hoặc độ dài câu trả lời. Thời gian LLM gồm xử lý đầu vào (prefill) và sinh token tự hồi quy (decode), chưa tách riêng.", ""]
    if "answer_comparisons" in report:
        comparisons = report["answer_comparisons"]
        matches = sum(row["exact_match"] for row in comparisons)
        lines += ["### Đối chiếu câu trả lời", "", f"Số câu trùng khớp hoàn toàn: **{matches}/{len(comparisons)}**. Đây là độ đồng thuận với baseline, không phải độ chính xác so với nhãn chuẩn. Số token được đếm trực tiếp từ chuỗi trả về, gồm token đặc biệt. Văn bản đầu ra mô hình bên dưới được giữ nguyên để đối chiếu.", ""]
        for row in comparisons:
            lines += [f"Lượt {row['repeat']}, ảnh `{row['frame']}`, chế độ khung STC `{row['stc_vision_mode']}`:", "",
                      "```text", "Không STC: " + row["baseline_answer"], "Có STC: " + row["stc_answer"], "```", ""]
    return "\n".join(lines) + "\n"


def save(path, report):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)
    path.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "wad_sample/images")
    parser.add_argument("--limit", type=int, default=9, help="Lấy N ảnh JPG đầu theo thứ tự; 0 là lấy tất cả")
    parser.add_argument("--guidance-every", type=int, default=9)
    parser.add_argument("--device", default="cpu", help="Thiết bị: cpu hoặc cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--cache-interval", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=2, help="Mỗi chế độ làm nóng vision/tracking và sinh một câu trả lời trước khi đo")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-stages", action="store_true", help="Đo riêng từng công đoạn để phân tích STC và thời gian từng module")
    parser.add_argument("--online", action="store_true", help="Cho phép tải từ Hugging Face; mặc định chạy ngoại tuyến")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/system_benchmark.json")
    args = parser.parse_args()
    if (args.limit < 0 or args.guidance_every < 1 or args.cache_interval < 2
            or args.repeats < 1 or args.warmup_frames < 2 or args.threads < 1):
        parser.error("Require limit >= 0, guidance-every/repeats/threads >= 1, cache-interval/warmup-frames >= 2")
    if not math.isfinite(args.ratio) or not 0 < args.ratio < 1:
        parser.error("--ratio must be in (0, 1)")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be in [0, 2**32)")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json; a companion .md is written")
    paths = sorted(args.images_dir.glob("*.jpg"))
    paths = paths[:args.limit] if args.limit else paths
    if len(paths) < 2:
        parser.error("Need at least two consecutive JPGs to exercise STC")

    os.environ.update(USE_TF="0", USE_TORCH="1", STC_PATCH_VISION="0",
                      HF_HUB_OFFLINE="0" if args.online else "1",
                      TRANSFORMERS_OFFLINE="0" if args.online else "1")
    global torch, np, Image, StageTimings, stage_scope, enable_stc_cacher
    import torch
    import numpy as np
    from PIL import Image
    from pipeline_timing import StageTimings, stage_scope
    from Vision_Encoder.stc_cacher import enable_stc_cacher
    import main as backend

    args.device = torch.device(args.device)
    if args.device.type not in {"cpu", "cuda"}:
        parser.error("Only cpu and cuda devices are supported")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.dtype == "auto":
        args.dtype = "bfloat16" if args.device.type == "cuda" else "float32"
    backend.DEVICE = str(args.device)
    backend.DTYPE = getattr(torch, args.dtype)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "loading", "settings": {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in vars(args).items()},
        "inputs": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "runs": {"baseline": [], "stc": []},
        "notes": ["Same resident full system; no model reloading between modes.",
                  "Every frame goes through tracking and vision in both modes; guidance schedule is identical.",
                  "Headline totals include disk decoding and CPU/GPU transfers; exclude reset, loading and warmup.",
                  "Baseline is streaming without STC, not the old final-frame-only demo.",
                  "No HTTP, browser, audio, navigation ground truth or per-token LLM timing.",
                  "This benchmark does not replace selective-100% correctness validation in benchmark_stc.py."],
    }
    save(args.output, report)
    runtime = None
    try:
        print("Đang nạp toàn bộ hệ thống một lần (ghi thời gian riêng)...", flush=True)
        start = time.perf_counter()
        runtime = backend.get_pipeline()
        synchronize(args.device)
        report["model_load_seconds"] = time.perf_counter() - start
        model = runtime.model_bundle[0]
        report["environment"] = {
            "torch": str(torch.__version__), "platform": platform.platform(),
            "device": str(args.device), "dtype": args.dtype,
            "gpu": torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else None,
            "base_model": backend.BASE_MODEL, "checkpoint": str(backend.CHECKPOINT),
            "model_revision": getattr(model.config, "_commit_hash", None),
            "prompt": backend.PROMPT, "do_sample": False, "max_new_tokens": 128,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                              ("main.py", "streaming_pipeline.py", "pipeline_timing.py", "Vision_Encoder/stc_cacher.py")},
        }
        report["status"] = "warming"
        report["warmups"] = {}
        with torch.inference_mode():
            for mode in ("baseline", "stc"):
                print(f"Làm nóng, không tính vào kết quả: {mode}", flush=True)
                report["warmups"][mode] = replay(runtime, paths[:args.warmup_frames], args, mode, warmup=True)
                save(args.output, report)
            torch.set_num_threads(args.threads)
            report["environment"]["measured_torch_threads"] = torch.get_num_threads()
            report["status"] = "measuring"
            for repeat in range(args.repeats):
                order = ("baseline", "stc") if repeat % 2 == 0 else ("stc", "baseline")
                for mode in order:
                    print(f"Đo toàn hệ thống, lượt {repeat + 1}/{args.repeats}: {mode}", flush=True)
                    result = replay(runtime, paths, args, mode)
                    result.update(repeat=repeat + 1, order_in_repeat=list(order))
                    report["runs"][mode].append(result)
                    save(args.output, report)
            report["summary"] = summarize(report["runs"])
            report["answer_comparisons"] = compare_answers(report["runs"])
            save(args.output, report)
            if args.profile_stages:
                report["status"] = "profiling_stages"
                report["profiles"] = {}
                for mode in ("baseline", "stc"):
                    print(f"Đo riêng từng công đoạn: {mode}", flush=True)
                    report["profiles"][mode] = replay(runtime, paths, args, mode, diagnostic=True)
                    save(args.output, report)
                report["stage_summary"] = {mode: summarize_stages(run) for mode, run in report["profiles"].items()}
        report["status"] = "complete"
        save(args.output, report)
        print(markdown(report), flush=True)
        print(f"JSON: {args.output.resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save(args.output, report)
        raise
    finally:
        if runtime is not None:
            runtime.stage_timings = None
            runtime.reset_stream()
            if runtime.stc_cache is not None:
                runtime.stc_cache.remove()
                runtime.stc_cache = None


if __name__ == "__main__":
    raise SystemExit(main())
