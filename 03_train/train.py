# Databricks notebook source
"""train.py — YOLOv8 training + Unity Catalog model registration.

Pipeline position: step 03 (train). This notebook/script trains a YOLOv8
detector on the volleyball-player dataset prepared earlier in the pipeline,
tracks the run with MLflow, and registers the resulting model into Unity
Catalog as a pyfunc wrapping the Ultralytics weights.

What it does, in order:
  1. Read job parameters from ``dbutils.widgets``.
  2. Point MLflow at the Unity Catalog model registry and at the per-user
     experiment ``/Users/{user}/yolo-cv``.
  3. Read the dataset manifest from
     ``/Volumes/{catalog}/{schema}/dataset/data.yaml``.
  4. Inside a single MLflow run: log all widgets as params, train the model,
     log validation metrics (mAP50, mAP50-95, precision, recall) and the
     confusion-matrix artifact.
  5. Register the trained model into Unity Catalog as a
     ``mlflow.pyfunc`` model named ``{catalog}.{schema}.{model_name}`` that
     wraps the best ``.pt`` weights via the sibling ``YOLOPyfunc`` class.

Conventions enforced here (project-wide):
  - Unity Catalog three-part namespace for every registered object.
  - Databricks Runtime ML 15.x compatible.
  - Zero hardcoded paths/credentials: everything via ``dbutils.widgets`` or the
    ``# CONFIGURE`` top-of-file constants.
  - UC registry (``databricks-uc``) + UC aliases (no legacy Staging/Production).

This file is a standalone Databricks script/notebook.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pandas as pd
import yaml
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Import the pyfunc wrapper from the sibling module ``yolo_pyfunc.py``.
#
# In a Databricks notebook the working directory is not guaranteed to be on
# ``sys.path``, so we add this file's directory before importing. The same
# ``yolo_pyfunc.py`` file is shipped with the logged model via ``code_paths``
# below, which keeps training and inference on the identical implementation.
# -----------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from yolo_pyfunc import YOLOPyfunc  # noqa: E402  (path adjusted above)

# -----------------------------------------------------------------------------
# CONFIGURE — top-of-file constants. No hardcoded credentials; these are
# project-level toggles that rarely change and are safe to keep in source.
# -----------------------------------------------------------------------------
REGISTRY_URI: str = "databricks-uc"  # CONFIGURE: UC model registry (do not use legacy stages)
EXPERIMENT_ROOT: str = "/Users"  # CONFIGURE: parent folder for the per-user MLflow experiment
EXPERIMENT_NAME: str = "yolo-cv"  # CONFIGURE: experiment leaf name -> /Users/{user}/yolo-cv
PROJECT_TAG: str = "volei-tactical"  # CONFIGURE: MLflow project tag (canonical project name)
ARTIFACT_PATH: str = "model"  # CONFIGURE: MLflow artifact sub-path for the logged pyfunc model
TRAIN_OUTPUT_ROOT: str = "/tmp/yolo_train"  # CONFIGURE: local scratch dir for Ultralytics run outputs

# Pinned floors for the model's serving environment. Mirrors requirements.txt;
# these are the packages the pyfunc needs at inference time.
PIP_REQUIREMENTS: List[str] = [  # CONFIGURE: pip requirements baked into the logged model
    "ultralytics>=8.2.0",
    "opencv-python-headless>=4.9.0",
    "mlflow>=2.14.0",
]

# Mapping from Ultralytics' ``results_dict`` keys to the clean metric names we
# want in MLflow. Robust to missing keys (some keys depend on the task/version).
_METRIC_NAME_MAP: Dict[str, str] = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


def get_widget_value(name: str, default: str) -> str:
    """Return a Databricks widget value, creating the widget if needed.

    Creating the widget with a default makes the script runnable both as a
    parameterized Databricks Job and interactively in a notebook.

    Args:
        name: Widget name.
        default: Default value used when the widget has no explicit value.

    Returns:
        The widget's current string value.
    """
    dbutils.widgets.text(name, default)  # type: ignore[name-defined]  # noqa: F821
    return dbutils.widgets.get(name)  # type: ignore[name-defined]  # noqa: F821


def resolve_current_user(spark_session: "SparkSession") -> str:  # type: ignore[name-defined]  # noqa: F821
    """Resolve the current Databricks user via Spark SQL.

    Args:
        spark_session: Active Spark session.

    Returns:
        The current user identifier (typically an email address).
    """
    return spark_session.sql("select current_user()").first()[0]


def read_dataset_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load the Ultralytics dataset manifest (``data.yaml``).

    Args:
        yaml_path: Absolute path to ``data.yaml`` on a UC Volume.

    Returns:
        The parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset manifest not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_metrics(results_dict: Dict[str, float]) -> Dict[str, float]:
    """Map Ultralytics result keys to clean metric names, skipping missing keys.

    Args:
        results_dict: ``results.results_dict`` produced by ``model.train``.

    Returns:
        A dictionary of ``{clean_metric_name: value}`` containing only the
        metrics that were actually present and numeric.
    """
    metrics: Dict[str, float] = {}
    for ultralytics_key, clean_name in _METRIC_NAME_MAP.items():
        value = results_dict.get(ultralytics_key)
        if value is not None:
            try:
                metrics[clean_name] = float(value)
            except (TypeError, ValueError):
                # Non-numeric value — skip rather than fail the run.
                continue
    return metrics


def main() -> None:
    """Train YOLOv8, log to MLflow, and register the model in Unity Catalog."""
    # --- 1. Read job parameters from widgets -------------------------------
    catalog = get_widget_value("catalog", "main")
    schema = get_widget_value("schema", "volei_tactical")
    model_name = get_widget_value("model_name", "yolo_volei")
    epochs = int(get_widget_value("epochs", "50"))
    imgsz = int(get_widget_value("imgsz", "640"))
    batch_size = int(get_widget_value("batch_size", "16"))
    base_model = get_widget_value("base_model", "yolov8n.pt")

    # --- 2. MLflow: UC registry + per-user experiment ----------------------
    mlflow.set_registry_uri(REGISTRY_URI)
    user = resolve_current_user(spark)  # type: ignore[name-defined]  # noqa: F821
    experiment_path = f"{EXPERIMENT_ROOT}/{user}/{EXPERIMENT_NAME}"
    mlflow.set_experiment(experiment_path)

    # --- 3. Read the dataset manifest from the UC Volume -------------------
    yaml_path = f"/Volumes/{catalog}/{schema}/dataset/data.yaml"
    _ = read_dataset_yaml(yaml_path)  # validate the manifest exists and parses

    # Fully-qualified UC model name (three-part namespace).
    registered_model_name = f"{catalog}.{schema}.{model_name}"

    # --- 4. Train inside a single MLflow run -------------------------------
    with mlflow.start_run() as run:
        # Tag the run with the canonical project metadata.
        mlflow.set_tags({"project": PROJECT_TAG, "base_model": base_model})

        # Log every widget as a param for full reproducibility.
        mlflow.log_params(
            {
                "catalog": catalog,
                "schema": schema,
                "model_name": model_name,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch_size": batch_size,
                "base_model": base_model,
            }
        )

        # Train the YOLOv8 model. ``project``/``name`` control where Ultralytics
        # writes its run outputs (weights, plots, confusion matrix).
        run_name = run.info.run_id
        model = YOLO(base_model)
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=TRAIN_OUTPUT_ROOT,
            name=run_name,
        )

        # Log validation metrics, robust to keys missing across versions/tasks.
        metrics = extract_metrics(getattr(results, "results_dict", {}) or {})
        if metrics:
            mlflow.log_metrics(metrics)

        # Resolve the Ultralytics output directory for this run.
        save_dir = Path(results.save_dir)

        # Log the confusion-matrix plot if Ultralytics produced one.
        confusion_matrix_path = save_dir / "confusion_matrix.png"
        if confusion_matrix_path.exists():
            mlflow.log_artifact(str(confusion_matrix_path))

        # Resolve the best weights checkpoint produced by training.
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.exists():
            # Fall back to the last checkpoint if ``best.pt`` is absent.
            best_weights = save_dir / "weights" / "last.pt"
        if not best_weights.exists():
            raise FileNotFoundError(
                f"No trained weights found under: {save_dir / 'weights'}"
            )

        # --- 5. Register the model into Unity Catalog as a pyfunc ----------
        # The model wraps the ``.pt`` weights via the sibling ``YOLOPyfunc``
        # class. ``code_paths`` ships ``yolo_pyfunc.py`` so the exact same
        # implementation is used at inference time (training/serving parity).
        #
        # Best-effort model signature so the registered model self-documents its
        # IO contract (input: a ``path`` column; output: frame_id/boxes/confs/
        # classes). Signature inference must never break a successful run.
        input_example = pd.DataFrame(
            {"path": [f"/Volumes/{catalog}/{schema}/dataset/frames/example__000000.jpg"]}
        )
        signature = None
        try:
            from mlflow.models import infer_signature

            output_example = pd.DataFrame(
                [
                    {
                        "frame_id": "example__000000",
                        "boxes": [
                            {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "conf": 0.9, "cls": 0}
                        ],
                        "confs": [0.9],
                        "classes": [0],
                    }
                ]
            )
            signature = infer_signature(input_example, output_example)
        except Exception:  # noqa: BLE001 — signature is best-effort, never break training
            signature = None

        mlflow.pyfunc.log_model(
            artifact_path=ARTIFACT_PATH,
            python_model=YOLOPyfunc(),
            artifacts={"weights": str(best_weights)},
            code_paths=[str(_THIS_DIR / "yolo_pyfunc.py")],
            registered_model_name=registered_model_name,
            pip_requirements=PIP_REQUIREMENTS,
            signature=signature,
            input_example=input_example,
        )

        print(
            f"Logged and registered model '{registered_model_name}' "
            f"from run '{run.info.run_id}'. "
            f"Metrics: {metrics if metrics else 'none captured'}."
        )


# In a Databricks notebook the module name is not ``__main__``; run on import.
main()
