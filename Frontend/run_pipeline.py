"""Select a demo sequence without editing main.py or replacing sample files."""
import argparse
import importlib.util
from pathlib import Path
import sys


def run_pipeline(pipeline_path, images_dir):
    pipeline_path = Path(pipeline_path).resolve()
    # main.py's imports must resolve as they do when run from the project root.
    sys.path.insert(0, str(pipeline_path.parent))
    spec = importlib.util.spec_from_file_location("navsvlm_demo_pipeline", pipeline_path)
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    # Only the input directory changes, inside this subprocess. All checkpoint,
    # preprocessing, tracking, prompt and generation settings remain main.py's.
    pipeline.IMAGES_DIR = Path(images_dir).resolve()
    pipeline.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    args = parser.parse_args()
    run_pipeline(args.pipeline, args.images_dir)
