"""Local demo launcher with one persistent inference worker and resident models."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent
PIPELINE = ROOT / "main.py"
IMAGES = ROOT / "wad_sample" / "images"
SCENES = {
    "wad_sample": IMAGES,
    "wad_sample_2": (ROOT / "wad_sample_2" / "images"
                     if (ROOT / "wad_sample_2" / "images").is_dir()
                     else ROOT / "wad_sample" / "wad_sample_2" / "images"),
}
ASSETS = {"/": ("index.html", "text/html"), "/styles.css": ("styles.css", "text/css"),
          "/app.js": ("app.js", "text/javascript")}
EVENT_PREFIX = "NAVSVLM_EVENT "


class DemoHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let two launchers serve the same port.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def sample_frames(scene_id="wad_sample"):
    # Match directory replay ordering and the demo's nine-frame limit.
    return sorted(SCENES[scene_id].glob("*.jpg"))[:9]


def scene_description(scene_id, index):
    return {"id": scene_id, "title": f"Scene {index + 1}",
            "folder": SCENES[scene_id].relative_to(ROOT).as_posix(),
            "frames": [{"name": frame.name, "url": f"/frames/{scene_id}/{number}"}
                       for number, frame in enumerate(sample_frames(scene_id))]}


class Runner:
    def __init__(self, offline=True):
        self.offline = offline
        self.lock = threading.Lock()
        self.process = None
        self.started = None
        self.finished = None
        self.status = "idle"
        self.answer = ""
        self.error = ""
        self.logs = []
        self.scene_id = None
        self.worker_ready = False
        self.frames_done = 0
        self.frames_total = 0
        self.request_id = 0
        self.closed = False
        self._threads = ()

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "answer": self.answer, "error": self.error,
                    "scene_id": self.scene_id,
                    "worker_ready": self.worker_ready,
                    "frames_done": self.frames_done, "frames_total": self.frames_total,
                    "logs": list(self.logs), "elapsed": round(
                        (self.finished or time.monotonic()) - self.started, 1
                    ) if self.started is not None else 0}

    def start(self, scene_id="wad_sample"):
        if scene_id not in SCENES:
            raise ValueError("Unknown sample scene.")
        with self.lock:
            if self.closed:
                raise RuntimeError("Runner has been closed")
            if self.status == "running":
                return False
            self.status, self.answer, self.error = "running", "", ""
            self.logs = []
            self.frames_done, self.frames_total = 0, 0
            self.request_id += 1
            self.scene_id = scene_id
            self.started, self.finished = time.monotonic(), None
            try:
                if self.process is None or self.process.poll() is not None:
                    if self.process is not None and self.process.stdin is not None:
                        self.process.stdin.close()
                    self.worker_ready = False
                    self.process = subprocess.Popen(
                        [sys.executable, "-u", str(FRONTEND / "run_pipeline.py"),
                         "--pipeline", str(PIPELINE), "--worker"], cwd=str(ROOT),
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", errors="replace", bufsize=1,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                             "HF_HUB_OFFLINE": "1" if self.offline else "0",
                             "TRANSFORMERS_OFFLINE": "1" if self.offline else "0"},
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self._threads = (
                        threading.Thread(target=self._collect, args=(self.process,), daemon=True),
                        threading.Thread(target=self._collect_logs, args=(self.process,), daemon=True),
                    )
                    for thread in self._threads:
                        thread.start()
                self.process.stdin.write(json.dumps({
                    "id": self.request_id, "command": "run", "images_dir": str(SCENES[scene_id]), "limit": 9,
                }) + "\n")
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                self.status, self.error = "error", str(exc)
                self.finished = time.monotonic()
                self.worker_ready = False
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                return True
            return True

    def _collect_logs(self, process):
        # Keep progress bars / partial log lines off the JSON protocol pipe.
        with process.stderr:
            for line in process.stderr:
                with self.lock:
                    if process is self.process and not self.closed:
                        self.logs.append(line.rstrip())
                        self.logs = self.logs[-150:]

    def _collect(self, process):
        try:
            with process.stdout:
                for line in process.stdout:
                    line = line.rstrip()
                    with self.lock:
                        if process is not self.process or self.closed:
                            continue
                        if line.startswith(EVENT_PREFIX):
                            try:
                                event = json.loads(line[len(EVENT_PREFIX):])
                            except ValueError:
                                event = {}
                            if event.get("id") != self.request_id:
                                continue
                            kind = event.get("event")
                            if kind == "ready":
                                self.worker_ready = True
                            elif kind == "progress":
                                self.frames_done = event["frames_done"]
                                self.frames_total = event["frames_total"]
                            elif kind in {"complete", "error"} and self.status == "running":
                                self.answer = event.get("answer", "") if kind == "complete" else ""
                                self.status = "complete" if self.answer else "error"
                                self.error = event.get("error", "") if kind == "error" else ""
                                if self.status == "error" and not self.error:
                                    self.error = "The pipeline finished without generated guidance."
                                if self.error:
                                    self.logs.append(f"Error: {self.error}")
                                    self.logs = self.logs[-150:]
                                self.finished = time.monotonic()
                        else:
                            self.logs.append(line)
                            self.logs = self.logs[-150:]
            code = process.wait()
            with self.lock:
                if process is self.process and not self.closed:
                    self.worker_ready = False
                    if self.status == "running":
                        self.answer = ""
                        self.status = "error"
                        self.error = f"Worker exited with code {code}. Retry to start a new worker."
                        self.finished = time.monotonic()
        except Exception as exc:
            with self.lock:
                if process is self.process and not self.closed:
                    self.status, self.error = "error", str(exc)
                    self.finished = time.monotonic()
                    self.worker_ready = False
                    if process.poll() is None:
                        process.terminate()

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
            self.worker_ready = False
            if self.status == "running":
                self.status, self.error = "error", "Worker stopped."
                self.finished = time.monotonic()
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                    process.stdin.flush()
                except (OSError, ValueError):
                    process.terminate()
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                for thread in self._threads:
                    thread.join(timeout=1)


class Handler(BaseHTTPRequestHandler):
    def _local_request(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _send(self, code, data, content_type="application/json"):
        if content_type == "application/json":
            data = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type != "image/jpeg" else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if not self._local_request():
            return self._send(403, {"error": "Local access only."})
        path = urlsplit(self.path).path
        if path == "/api/session":
            scenes = [scene_description(scene_id, index) for index, scene_id in enumerate(SCENES)]
            return self._send(200, {"scenes": scenes, "frames": scenes[0]["frames"]})
        if path == "/api/status":
            return self._send(200, self.server.runner.snapshot())
        if path in ASSETS:
            filename, mime = ASSETS[path]
            return self._send(200, (FRONTEND / filename).read_bytes(), mime)
        if path.startswith("/frames/"):
            parts = path.removeprefix("/frames/").split("/")
            if len(parts) == 1:
                scene_id, index = "wad_sample", parts[0]
            elif len(parts) == 2 and parts[0] in SCENES:
                scene_id, index = parts
            else:
                return self._send(404, {"error": "Not found."})
            frames = sample_frames(scene_id)
            if index.isdigit() and len(index) <= 2 and int(index) < len(frames):
                return self._send(200, frames[int(index)].read_bytes(), "image/jpeg")
        self._send(404, {"error": "Not found."})

    def do_POST(self):
        if not self._local_request() or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
            return self._send(403, {"error": "Open the demo from its local address to run the pipeline."})
        if urlsplit(self.path).path != "/api/run":
            return self._send(404, {"error": "Not found."})
        scene_id = parse_qs(urlsplit(self.path).query).get("scene", ["wad_sample"])[0]
        if scene_id not in SCENES:
            return self._send(400, {"error": "Unknown sample scene."})
        if not sample_frames(scene_id):
            return self._send(400, {"error": f"No sample JPG frames found for {scene_id}."})
        if not self.server.runner.start(scene_id):
            return self._send(409, {"error": "A run is already in progress."})
        self._send(202, self.server.runner.snapshot())

    def log_message(self, format, *args):
        # Keep the terminal quiet while the UI polls status.
        if args and str(args[1]) not in {"200", "202"}:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--online", action="store_true",
                        help="Allow Hugging Face downloads and update checks (default: cached files only).")
    args = parser.parse_args()
    try:
        server = DemoHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Cannot listen on port {args.port}: {exc}\n"
                       "Stop the existing launcher or choose another --port.\n")
    server.runner = Runner(offline=not args.online)
    print(f"NavSVLM demo: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Pipeline interpreter: {sys.executable}", flush=True)
    print(f"Hugging Face: {'online' if args.online else 'offline (cached files only)'}", flush=True)
    print("Models load on the first request and stay in memory until Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.runner.close()


if __name__ == "__main__":
    main()
