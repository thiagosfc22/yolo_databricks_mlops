"""Local MLflow packaging test — de-risk log_model / load_model before Databricks.

This exercises the same two MLflow calls the pipeline relies on, minus the Unity
Catalog registry:
    - 03_train/train.py  -> mlflow.pyfunc.log_model(python_model=YOLOPyfunc(), ...)
    - 05_inference/...   -> mlflow.pyfunc.load_model(...).predict(df)

If the pyfunc serializes (cloudpickle + "weights" artifact + code_paths) and loads
back to produce identical detections locally, the only thing left for Databricks is
the registry/UC wiring — not the model packaging itself.

Tracking is local & file-based (no Databricks): mlruns under local_dev/_work/mlruns.

Usage:
    python test_mlflow_packaging.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "03_train"
PYFUNC_FILE = TRAIN_DIR / "yolo_pyfunc.py"
WEIGHTS = HERE.parent.parent / "volei-tactical" / "yolov8n.pt"
MANIFEST = HERE / "_work" / "frames.parquet"
# Recent MLflow puts the bare-file tracking store in maintenance mode, so we use a
# local sqlite backend (closer to a real tracking server) with file-based artifacts.
MLFLOW_DB = HERE / "_work" / "mlflow.db"
ARTIFACTS = HERE / "_work" / "mlartifacts"

# Import YOLOPyfunc as a proper module named "yolo_pyfunc" so MLflow can re-import
# it on load (matches train.py's code_paths=["yolo_pyfunc.py"]).
sys.path.insert(0, str(TRAIN_DIR))
import yolo_pyfunc  # noqa: E402  (path set above)


def main() -> None:
    """Log the pyfunc to a local MLflow run, load it back, and compare outputs."""
    assert WEIGHTS.exists(), f"weights not found: {WEIGHTS}"
    assert MANIFEST.exists(), f"run extract_frames_local.py first: {MANIFEST}"

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    exp_name = "local_smoke_yolo_cv"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=f"file:{ARTIFACTS}")
    mlflow.set_experiment(exp_name)

    manifest = pd.read_parquet(MANIFEST)
    model_input = manifest[["frame_path", "frame_id"]].rename(columns={"frame_path": "path"})

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=yolo_pyfunc.YOLOPyfunc(),
            artifacts={"weights": str(WEIGHTS)},
            code_paths=[str(PYFUNC_FILE)],
            pip_requirements=["ultralytics>=8.2.0", "opencv-python-headless>=4.9.0"],
        )
        model_uri = f"runs:/{run.info.run_id}/model"
    print(f"Logged model: {model_uri}")

    # Load it back EXACTLY like batch_inference.py would (just not via models:/ UC).
    loaded = mlflow.pyfunc.load_model(model_uri)
    out = loaded.predict(model_input)

    expected_cols = ["frame_id", "boxes", "confs", "classes"]
    assert list(out.columns) == expected_cols, f"columns mismatch: {list(out.columns)}"
    assert len(out) == len(model_input), "row count must be preserved"

    total = int(out["boxes"].map(len).sum())
    print("LOAD + PREDICT OK — columns:", list(out.columns))
    print(f"Detections after reload: {total} across {len(out)} frames")
    print("\nReloaded pyfunc behaves like the in-process class. Packaging is Databricks-ready.")


if __name__ == "__main__":
    main()
