# NavSVLM demo

A small, local browser interface for the navigation pipeline. The launcher keeps
one inference worker alive and reuses its loaded models. No npm build,
external fonts, or additional Python packages are required for the interface.

From the repository root in PowerShell, use the Python environment that already
runs `main.py`:

```powershell
.\.venv\Scripts\python.exe Frontend\server.py
```

Open **http://127.0.0.1:8000**. If your working pipeline uses a different Python
environment, use that interpreter instead. To choose another port:

```powershell
.\.venv\Scripts\python.exe Frontend\server.py --port 8080
```

The demo uses **cached Hugging Face files by default**. The launcher sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the inference subprocess before
imports, avoiding online update checks and connection retries when models are
already cached. It does not change your global environment or backend code.
If required files are missing, allow downloads explicitly with:

```powershell
.\.venv\Scripts\python.exe Frontend\server.py --online
```

Online mode needs a working connection to Hugging Face. Restart the launcher
after changing modes. Offline mode cannot supply missing model files.

## Try the demo

1. Use the **left/right arrows** beside Scene input to switch between sequences.
   Browse the thumbnails or press play to preview the selected nine-frame sequence.
   Each press of Play starts from the first frame and repeats the sequence nine
   times, then stops on the final frame. Press Pause to stop playback early.
2. Click **Generate guidance** to run the existing pipeline on the selected scene
   with the same Python interpreter as the launcher, from the repository root.
3. Guidance is automatically read aloud when a run finishes. Use **Stop reading**
   to stop the voice, **Read aloud** to replay it, or **Copy text**. Expand
   **Run log** for model output and errors. Refreshing a completed result does not
   replay it. If your browser blocks automatic playback, select **Read aloud**.

Scene 1 uses `wad_sample/images`. Scene 2 uses `wad_sample/wad_sample_2/images`
(or `wad_sample_2/images` if the second sample is placed at the repository root).
The preview and inference use the same first nine alphabetically sorted JPGs in
the selected folder. Choosing a thumbnail only changes the preview; inference
uses the selected sequence and its final frame. Scene switching stops playback
and speech, and hides any guidance belonging to a different scene. Arrows are
disabled during generation. No upload,
camera capture, prompt editing, simulated predictions, or invented detection
overlays are included in this first demo.

The launcher serves only the frontend and the selected sample images on localhost.
It starts `Frontend/run_pipeline.py --worker` on the first request. That worker
loads InternViT/LLM, Q-Former, the tracking encoder and YOLO once, then accepts
subsequent requests over JSON lines. Results and per-frame progress use structured
events; model logs use a separate pipe. Each click replays the selected sequence:
tracking and STC state reset, but weights stay loaded. No files or checkpoints
are replaced. Hugging Face downloads require `--online`.
The first run can take several minutes. Later runs reuse the same worker until
the launcher stops. If the worker crashes, the current request reports an error;
the next request starts a new worker and loads models again.

Keep the launcher terminal open. Refreshing the page reconnects to the current
run; only one run is allowed at a time. Press **Ctrl+C** in the terminal to stop
the launcher and any active inference subprocess. Results are kept in memory
until a new run or launcher restart. Read-aloud uses your browser's available
speech voices and requires browser support.

If inference fails, inspect **Run log** and confirm that `main.py` works with the
same interpreter. The interface can be opened without loading model weights;
generating guidance requires the existing pipeline dependencies and checkpoints.

## Verify the launcher

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s Frontend -p "test_*.py" -v
```

These tests use small temporary workers to check serving, sample ordering,
model-runtime reuse, request errors, crash recovery and shutdown. They do not
load model checkpoints.

## Streaming video or camera

From the repository root, using your pipeline Python environment:

```powershell
$env:STC_PATCH_VISION="1"
$env:STC_UPDATE_TOKEN_RATIO="0.25"
$env:STC_CACHE_INTERVAL="4"
python main.py --video path\to\video.mp4 --guidance-every 9
# Or: python main.py --camera 0 --guidance-every 9
```

The loop reads one frame at a time and generates guidance every N frames and on
the final partial group. Tracking and STC state continue across guidance outputs.
Ctrl+C releases the capture. Inference is sequential; this is not a realtime
latency guarantee. The browser remains a sample replay interface, not a camera UI.
Set the same STC environment variables before starting `Frontend/server.py` to
enable caching there. Restart a running launcher to apply environment changes.

Python integrations can use `runtime = main.get_pipeline()` and repeatedly call
`runtime.process_frame(rgb_uint8, stream_id="video-1", frame_id=index, generate=True)`.
Use increasing frame IDs; change `stream_id` or call `runtime.reset_stream()` for
a new video, scene cut or seek. Only temporal state is reset. Do not start a new
Python process per frame; model residency is per process. STC is optional and
disabled unless `STC_PATCH_VISION=1`.

Runtime lifecycle checks with fake models (run manually):

```powershell
python -m unittest discover -s tests -p "test_streaming_pipeline.py" -v
```
