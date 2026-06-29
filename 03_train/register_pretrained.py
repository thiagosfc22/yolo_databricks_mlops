# Databricks notebook source
"""Register a PRETRAINED YOLOv8 model into Unity Catalog (no-labels fast path).

Pipeline position
-----------------
Alternative to ``03_train/train.py`` for when there is NO labeled dataset yet.
Instead of training, it wraps the off-the-shelf Ultralytics weights (e.g.
``yolov8n.pt``, COCO ``person`` detection) in the same :class:`YOLOPyfunc`,
logs it to MLflow, registers it in Unity Catalog as
``{catalog}.{schema}.{model_name}`` and assigns the ``@champion`` alias — so
stage ``05_inference`` can run end-to-end immediately.

Swap this for ``train.py`` + ``promote_model.py`` once a labeled, court-filtered
dataset exists. The downstream contract (the registered pyfunc + alias) is
identical, so nothing else in the pipeline changes.

Why this is useful
------------------
The generic COCO model detects every person (players, bench, referee, crowd).
That is enough to exercise the full MLOps path (ingest -> features -> registry
-> batch inference -> monitoring) and to make the "why train a custom model"
gap visible before investing in annotation.
"""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Import the SAME pyfunc wrapper used by training/inference (training/serving
# parity). The working dir is not guaranteed on sys.path inside a notebook.
# -----------------------------------------------------------------------------
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from yolo_pyfunc import YOLOPyfunc  # noqa: E402  (path adjusted above)

# -----------------------------------------------------------------------------
# CONFIGURE — project-level constants (no secrets).
# -----------------------------------------------------------------------------
REGISTRY_URI: str = "databricks-uc"  # CONFIGURE: UC model registry
EXPERIMENT_ROOT: str = "/Users"  # CONFIGURE: parent folder for the per-user experiment
EXPERIMENT_NAME: str = "yolo-cv"  # CONFIGURE: -> /Users/{user}/yolo-cv
PROJECT_TAG: str = "volei-tactical"  # CONFIGURE: canonical project tag
ARTIFACT_PATH: str = "model"  # CONFIGURE: MLflow artifact sub-path
PIP_REQUIREMENTS = [  # CONFIGURE: serving-env requirements baked into the model
    "ultralytics>=8.2.0",
    "opencv-python-headless>=4.9.0",
    "mlflow>=2.14.0",
]

# -----------------------------------------------------------------------------
# Widgets — all Unity Catalog coordinates via dbutils (no hardcoded namespaces).
# -----------------------------------------------------------------------------
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")  # CONFIGURE
dbutils.widgets.text("schema", "volei_tactical", "Unity Catalog schema")  # CONFIGURE
dbutils.widgets.text("model_name", "yolo_volei", "UC registered model name")  # CONFIGURE
dbutils.widgets.text("base_model", "yolov8n.pt", "Pretrained Ultralytics weights")  # CONFIGURE
dbutils.widgets.text("alias", "champion", "UC alias to assign")  # CONFIGURE


def resolve_current_user() -> str:
    """Return the current Databricks user (for the per-user experiment path)."""
    return spark.sql("select current_user()").first()[0]  # type: ignore[name-defined]  # noqa: F821


def main() -> None:
    """Wrap pretrained weights as YOLOPyfunc, register in UC, set the alias."""
    catalog = dbutils.widgets.get("catalog").strip()
    schema = dbutils.widgets.get("schema").strip()
    model_name = dbutils.widgets.get("model_name").strip()
    base_model = dbutils.widgets.get("base_model").strip() or "yolov8n.pt"
    alias = dbutils.widgets.get("alias").strip() or "champion"

    if not catalog or not schema or not model_name:
        raise ValueError("Widgets 'catalog', 'schema' and 'model_name' are required.")

    registered_model_name = f"{catalog}.{schema}.{model_name}"

    # MLflow: UC registry + per-user experiment.
    mlflow.set_registry_uri(REGISTRY_URI)
    user = resolve_current_user()
    mlflow.set_experiment(f"{EXPERIMENT_ROOT}/{user}/{EXPERIMENT_NAME}")

    # Trigger Ultralytics to download the pretrained checkpoint, then log that .pt
    # as the model's "weights" artifact (same shape as a trained best.pt).
    # Resolve the ABSOLUTE path Ultralytics actually used — a bare relative name
    # ("yolov8n.pt") is resolved against the process CWD at log time, which is not
    # guaranteed to be the download dir on a Databricks cluster (FileNotFoundError).
    yolo = YOLO(base_model)  # downloads base_model if not present
    weights_path = str(getattr(yolo, "ckpt_path", None) or os.path.abspath(base_model))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not resolve weights path for '{base_model}': {weights_path}")

    # Best-effort signature so the registered model self-documents its IO.
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
                    "boxes": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "conf": 0.9, "cls": 0}],
                    "confs": [0.9],
                    "classes": [0],
                }
            ]
        )
        signature = infer_signature(input_example, output_example)
    except Exception:  # noqa: BLE001 — signature is best-effort
        signature = None

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {"project": PROJECT_TAG, "base_model": base_model, "model_type": "pretrained-coco"}
        )
        model_info = mlflow.pyfunc.log_model(
            artifact_path=ARTIFACT_PATH,
            python_model=YOLOPyfunc(),
            artifacts={"weights": weights_path},
            code_paths=[os.path.join(_THIS_DIR, "yolo_pyfunc.py")],
            registered_model_name=registered_model_name,
            pip_requirements=PIP_REQUIREMENTS,
            signature=signature,
            input_example=input_example,
        )

    # Assign the alias to the freshly registered version so 05_inference can load
    # it via models:/{catalog}.{schema}.{model_name}@{alias}.
    version = model_info.registered_model_version
    client = MlflowClient()
    client.set_registered_model_alias(name=registered_model_name, alias=alias, version=version)

    print(
        f"Registered pretrained '{base_model}' as '{registered_model_name}' "
        f"version {version} and set @{alias} (run {run.info.run_id})."
    )


# COMMAND ----------

main()
