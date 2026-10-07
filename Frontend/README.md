# NavSVLM demo

A small, local browser interface for the existing navigation pipeline. All demo
files live in `Frontend`; the backend and model code are unchanged. No npm build,
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

1. Browse the sample thumbnails or press play to preview the nine-frame sequence.
2. Click **Generate guidance** to execute the existing `main.py` with the same
   Python interpreter as the launcher, from the repository root.
3. Guidance is automatically read aloud when a run finishes. Use **Stop reading**
   to stop the voice, **Read aloud** to replay it, or **Copy text**. Expand
   **Run log** for model output and errors. Refreshing a completed result does not
   replay it. If your browser blocks automatic playback, select **Read aloud**.

The preview uses the same first nine alphabetically sorted JPG files in
`wad_sample/images` as the backend. Choosing a thumbnail only changes the preview;
inference always uses the full sample sequence and its final frame. No upload,
camera capture, prompt editing, simulated predictions, or invented detection
overlays are included in this first demo.

The launcher serves only the frontend and the selected sample images on localhost.
It starts a subprocess when requested and reads the existing `Generated text:`
output. Model loading, dependencies, GPU selection, and cache writes remain the
existing pipeline's behavior; Hugging Face downloads require `--online`.
The first run can take several minutes;
every run reloads models because `main.py` is a standalone script. Progress is
shown as elapsed time and real logs, since the backend has no progress API.

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

These tests use small temporary subprocesses to check serving, sample ordering,
run isolation, result parsing, and errors. They do not run the model.
