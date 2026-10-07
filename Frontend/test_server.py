"""Launcher integration tests with tiny subprocesses; no model weights are loaded."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import server


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.runner = server.Runner()
        self.httpd = server.DemoHTTPServer(("127.0.0.1", 0), server.Handler)
        self.httpd.runner = self.runner
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.httpd.server_port}"
        self.temp = tempfile.TemporaryDirectory()
        self.pipeline = Path(self.temp.name) / "pipeline.py"
        self.patcher = patch.object(server, "PIPELINE", self.pipeline)
        self.patcher.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.runner.close()
        self.thread.join(timeout=2)
        self.patcher.stop()
        self.temp.cleanup()

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=5)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        code, body = response.status, response.read()
        connection.close()
        return code, body

    def wait_for_result(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.runner.snapshot()
            if result["status"] != "running":
                return result
            time.sleep(0.02)
        self.fail("Subprocess did not finish")

    def test_serves_frontend_and_backend_sample_order(self):
        for path in ("/", "/styles.css", "/app.js"):
            self.assertEqual(self.request(path)[0], 200)
        code, body = self.request("/api/session")
        self.assertEqual(code, 200)
        frames = json.loads(body)["frames"]
        expected = sorted(server.IMAGES.glob("*.jpg"))[:9]
        self.assertEqual([f["name"] for f in frames], [p.name for p in expected])
        for frame in frames:
            code, data = self.request(frame["url"])
            self.assertEqual(code, 200)
            self.assertTrue(data.startswith(b"\xff\xd8"))

    def test_does_not_expose_repository_files(self):
        for path in ("/main.py", "/../main.py", "/frames/../main.py", "/frames/99"):
            self.assertEqual(self.request(path)[0], 404)
        self.assertEqual(self.request("/", headers={"Host": "untrusted.example"})[0], 403)

    def test_rejects_cross_origin_run(self):
        self.assertEqual(self.request("/api/run", "POST", {"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.runner.snapshot()["status"], "idle")

    def test_rejects_duplicate_listener(self):
        with self.assertRaises(OSError):
            duplicate = server.DemoHTTPServer(self.httpd.server_address, server.Handler)
            duplicate.server_close()

    def test_success_multiline_output_and_duplicate_run(self):
        self.pipeline.write_text("import time\nprint('Loading...')\ntime.sleep(0.5)\nprint('Generated text: Pause here.')\nprint('A pedestrian is ahead.')\n", encoding="utf-8")
        self.assertEqual(self.request("/api/run", "POST", {"Origin": self.origin})[0], 202)
        self.assertEqual(self.request("/api/run", "POST", {"Origin": self.origin})[0], 409)
        result = self.wait_for_result()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["answer"], "Pause here.\nA pedestrian is ahead.")
        self.assertIn("Loading...", result["logs"])
        self.assertEqual(json.loads(self.request("/api/status")[1])["answer"], result["answer"])

    def test_nonzero_exit_and_retry(self):
        self.pipeline.write_text("raise RuntimeError('test failure')\n", encoding="utf-8")
        self.runner.start()
        result = self.wait_for_result()
        self.assertEqual(result["status"], "error")
        self.assertIn("code 1", result["error"])
        self.assertTrue(any("test failure" in line for line in result["logs"]))
        self.pipeline.write_text("print('Generated text: Retry succeeded.')\n", encoding="utf-8")
        self.runner.start()
        self.assertEqual(self.wait_for_result()["answer"], "Retry succeeded.")

    def test_missing_output_is_not_success(self):
        self.pipeline.write_text("print('No guidance emitted')\n", encoding="utf-8")
        self.runner.start()
        result = self.wait_for_result()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")

    def test_empty_sample_prevents_launch(self):
        with patch.object(server, "sample_frames", return_value=[]):
            self.assertEqual(self.request("/api/run", "POST", {"Origin": self.origin})[0], 400)
        self.assertEqual(self.runner.snapshot()["status"], "idle")

    def test_offline_flags_reach_child_and_online_overrides_them(self):
        self.pipeline.write_text(
            "import os\nprint('Generated text:', os.environ['HF_HUB_OFFLINE'], os.environ['TRANSFORMERS_OFFLINE'])\n",
            encoding="utf-8",
        )
        for offline, expected in ((True, "1 1"), (False, "0 0")):
            with self.subTest(offline=offline):
                self.runner.offline = offline
                with patch.dict(server.os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
                    self.runner.start()
                self.assertEqual(self.wait_for_result()["answer"], expected)


if __name__ == "__main__":
    unittest.main()
