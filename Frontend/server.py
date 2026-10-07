"""Local demo launcher. Runs the existing main.py without importing or editing it."""
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
from urllib.parse import urlsplit

FRONTEND = Path(__file__).resolve().parent
ROOT = FRONTEND.parent
PIPELINE = ROOT / "main.py"
IMAGES = ROOT / "wad_sample" / "images"
ASSETS = {"/": ("index.html", "text/html"), "/styles.css": ("styles.css", "text/css"),
          "/app.js": ("app.js", "text/javascript")}


class DemoHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let two launchers serve the same port.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def sample_frames():
    # Match the unchanged backend's ordering and nine-frame limit.
    return sorted(IMAGES.glob("*.jpg"))[:9]


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

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "answer": self.answer, "error": self.error,
                    "logs": list(self.logs), "elapsed": round(
                        (self.finished or time.monotonic()) - self.started, 1
                    ) if self.started is not None else 0}

    def start(self):
        with self.lock:
            if self.status == "running":
                return False
            self.status, self.answer, self.error = "running", "", ""
            self.logs = []
            self.started, self.finished = time.monotonic(), None
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-u", str(PIPELINE)], cwd=str(ROOT),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                         "HF_HUB_OFFLINE": "1" if self.offline else "0",
                         "TRANSFORMERS_OFFLINE": "1" if self.offline else "0"},
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                self.status, self.error = "error", str(exc)
                self.finished = time.monotonic()
                return True
            threading.Thread(target=self._collect, args=(self.process,), daemon=True).start()
            return True

    def _collect(self, process):
        output = []
        collecting_answer = False
        try:
            with process.stdout:
                for line in process.stdout:
                    line = line.rstrip()
                    with self.lock:
                        self.logs.append(line)
                        self.logs = self.logs[-150:]
                    if line.startswith("Generated text:"):
                        collecting_answer = True
                        output = [line.partition("Generated text:")[2].strip()]
                    elif collecting_answer:
                        output.append(line)
            code = process.wait()
            with self.lock:
                self.answer = "\n".join(output).strip() if code == 0 else ""
                self.status = "complete" if code == 0 and self.answer else "error"
                if self.status == "error":
                    self.error = (f"Pipeline exited with code {code}. See the run log below."
                                  if code else "The pipeline finished without generated guidance. See the run log.")
                self.finished = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.status, self.error = "error", str(exc)
                self.finished = time.monotonic()

    def close(self):
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                process.terminate()
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


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
            frames = sample_frames()
            return self._send(200, {"frames": [
                {"name": frame.name, "url": f"/frames/{index}"}
                for index, frame in enumerate(frames)
            ]})
        if path == "/api/status":
            return self._send(200, self.server.runner.snapshot())
        if path in ASSETS:
            filename, mime = ASSETS[path]
            return self._send(200, (FRONTEND / filename).read_bytes(), mime)
        if path.startswith("/frames/"):
            index = path.removeprefix("/frames/")
            frames = sample_frames()
            if index.isdigit() and len(index) <= 2 and int(index) < len(frames):
                return self._send(200, frames[int(index)].read_bytes(), "image/jpeg")
        self._send(404, {"error": "Not found."})

    def do_POST(self):
        if not self._local_request() or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
            return self._send(403, {"error": "Open the demo from its local address to run the pipeline."})
        if urlsplit(self.path).path != "/api/run":
            return self._send(404, {"error": "Not found."})
        if not sample_frames():
            return self._send(400, {"error": "No sample JPG frames found in wad_sample/images."})
        if not self.server.runner.start():
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
    print("Press Ctrl+C to stop. Models load only when you click Generate guidance.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.runner.close()


if __name__ == "__main__":
    main()
