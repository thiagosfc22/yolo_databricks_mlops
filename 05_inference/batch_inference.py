# Databricks notebook source
"""Batch inference notebook/script for the volei-tactical YOLOv8 pipeline.

Pipeline position
-----------------
Step 05 (inference), executed AFTER:
    01_ingest          -> populates {catalog}.{schema}.frames
    02_feature_store   -> populates {catalog}.{schema}.frame_features
    03_train           -> logs the YOLOPyfunc model
    04_registry        -> registers the model in Unity Catalog and assigns the
                          @champion / @challenger aliases

What this file does
-------------------
Loads the Unity Catalog registered model by alias (default ``champion``) as an
``mlflow.pyfunc`` model, then runs distributed batch inference over every frame
in ``{catalog}.{schema}.frames`` that has NOT yet been scored. Detections are
written idempotently to ``{catalog}.{schema}.detections`` via a Delta ``MERGE``
keyed on ``frame_id``.

Spark-first contract
--------------------
All batch orchestration uses the Spark DataFrame API + Delta. pandas is used
ONLY inside the ``@pandas_udf``, which wraps the pyfunc model. The model is
loaded lazily once per worker process (via ``mlflow.pyfunc.load_model``, which
ships the model's logged ``code_paths``) and cached, rather than broadcasting a
driver-loaded instance that the workers could not unpickle.

pyfunc <-> inference contract (must match 03_train/yolo_pyfunc.py)
-----------------------------------------------------------------
- ``YOLOPyfunc.predict`` takes a ``pandas.DataFrame`` with a ``path`` column.
- It returns a ``pandas.DataFrame`` with columns:
      ``frame_id``, ``boxes``, ``confs``, ``classes``.
- ``boxes`` is, per frame, a list of structs
      ``{x1: float, y1: float, x2: float, y2: float, conf: float, cls: int}``.
- This notebook only consumes the ``boxes`` column and materializes it as the
  Spark type ``array<struct<x1:float,y1:float,x2:float,y2:float,conf:float,cls:int>>``.
"""

from __future__ import annotations

from typing import List

import mlflow
import pandas as pd
from delta.tables import DeltaTable
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.pandas.functions import pandas_udf

# -----------------------------------------------------------------------------
# Spark session (provided by the Databricks runtime; fetched defensively so the
# file also imports cleanly outside an attached notebook).
# -----------------------------------------------------------------------------
spark: SparkSession = SparkSession.builder.getOrCreate()

# MLflow must target the Unity Catalog model registry for models:/ + @alias URIs.
mlflow.set_registry_uri("databricks-uc")

# -----------------------------------------------------------------------------
# Widgets — ZERO hardcoded namespaces. All Unity Catalog coordinates come in via
# dbutils.widgets so this notebook is reusable across catalogs/schemas/models.
# -----------------------------------------------------------------------------
dbutils.widgets.text("catalog", "", "Unity Catalog catalog")  # CONFIGURE
dbutils.widgets.text("schema", "", "Unity Catalog schema")  # CONFIGURE
dbutils.widgets.text("model_name", "", "UC registered model name")  # CONFIGURE
dbutils.widgets.text("alias", "champion", "UC model alias to load")  # CONFIGURE

catalog: str = dbutils.widgets.get("catalog").strip()
schema: str = dbutils.widgets.get("schema").strip()
model_name: str = dbutils.widgets.get("model_name").strip()
alias: str = dbutils.widgets.get("alias").strip() or "champion"

if not catalog or not schema or not model_name:
    raise ValueError(
        "Widgets 'catalog', 'schema' and 'model_name' are required and must be "
        "non-empty (alias defaults to 'champion')."
    )

# -----------------------------------------------------------------------------
# Canonical fully-qualified identifiers (three-part UC namespace everywhere).
# -----------------------------------------------------------------------------
frames_table: str = f"{catalog}.{schema}.frames"
detections_table: str = f"{catalog}.{schema}.detections"
registered_model: str = f"{catalog}.{schema}.{model_name}"
model_uri: str = f"models:/{registered_model}@{alias}"

# -----------------------------------------------------------------------------
# Resolve the concrete model version backing the alias, so every detection row
# records the exact version that produced it (lineage / reproducibility).
# -----------------------------------------------------------------------------
client = MlflowClient()
model_version_info = client.get_model_version_by_alias(name=registered_model, alias=alias)
model_version: str = str(model_version_info.version)

print(f"Loading model '{model_uri}' (resolved to version {model_version})")

# -----------------------------------------------------------------------------
# Model loading strategy: load the pyfunc LAZILY ON EACH WORKER (cached per Python
# process), NOT by broadcasting a driver-loaded instance.
#
# Broadcasting a loaded pyfunc pickles the custom ``YOLOPyfunc`` class, which the
# executors cannot unpickle unless that class' code is already importable on them
# (it fails with ``ModuleNotFoundError: No module named 'yolo_pyfunc'``). Calling
# ``mlflow.pyfunc.load_model`` on the worker instead ships the model's logged
# ``code_paths``, so the class is always importable. The module-level cache keeps
# this to one load per worker process (not one per partition/batch).
# -----------------------------------------------------------------------------
_WORKER_MODEL: dict = {}

# Exact Spark type for the detections "boxes" column. Kept as a module-level
# constant because it is the binding part of the pyfunc <-> inference contract.
BOXES_RETURN_TYPE: str = (
    "array<struct<x1:float,y1:float,x2:float,y2:float,conf:float,cls:int>>"
)


@pandas_udf(BOXES_RETURN_TYPE)  # type: ignore[call-overload]
def detect_udf(paths: pd.Series) -> pd.Series:
    """Run YOLOv8 detection on a batch of frame paths.

    Vectorized (Pandas) UDF. For each input frame path it invokes the per-worker
    (lazily loaded, cached) pyfunc model and returns the per-frame detection structs.

    Parameters
    ----------
    paths:
        A ``pandas.Series`` of frame paths (the ``frame_path`` column). Each
        value is a string pointing at a frame image (e.g. a ``/Volumes/...``
        path).

    Returns
    -------
    pandas.Series
        One element per input row. Each element is a list of structs
        ``{x1, y1, x2, y2, conf, cls}`` (the ``boxes`` column produced by
        ``YOLOPyfunc.predict``), index-aligned with the input ``paths`` Series.
    """
    # Lazily load the model once per worker process (cached across batches). This
    # uses the model's logged code_paths, unlike a broadcast of a loaded instance.
    if "model" not in _WORKER_MODEL:
        _WORKER_MODEL["model"] = mlflow.pyfunc.load_model(model_uri)
    model = _WORKER_MODEL["model"]

    # Build the single-column DataFrame the pyfunc contract expects.
    request: pd.DataFrame = pd.DataFrame({"path": paths.to_numpy()})

    # YOLOPyfunc.predict -> DataFrame[frame_id, boxes, confs, classes].
    predictions: pd.DataFrame = model.predict(request)

    # We only need the "boxes" column here; re-index defensively against the
    # input so the output is row-aligned even if the model reset the index.
    boxes = predictions["boxes"].reset_index(drop=True)
    boxes.index = paths.index
    return boxes


def read_unprocessed_frames() -> DataFrame:
    """Return frames that still need inference.

    Reads ``{catalog}.{schema}.frames`` and, when the detections table already
    exists, removes every frame already scored using a LEFT ANTI JOIN on
    ``frame_id`` (idempotent / incremental processing). When the detections
    table does not yet exist, all frames are returned.

    Returns
    -------
    pyspark.sql.DataFrame
        Frames to score, carrying at least ``frame_id``, ``video_id`` and
        ``frame_path``.
    """
    frames: DataFrame = spark.table(frames_table).select(
        "frame_id", "video_id", "frame_path"
    )

    if spark.catalog.tableExists(detections_table):
        already_scored: DataFrame = spark.table(detections_table).select("frame_id")
        frames = frames.join(already_scored, on="frame_id", how="left_anti")

    return frames


def build_detections(frames: DataFrame) -> DataFrame:
    """Apply the detection UDF and attach inference-provenance columns.

    Parameters
    ----------
    frames:
        Frames to score (output of :func:`read_unprocessed_frames`).

    Returns
    -------
    pyspark.sql.DataFrame
        Columns: ``frame_id``, ``video_id``, ``boxes``, ``model_version``,
        ``inferred_at``, ``alias_used``.
    """
    return (
        frames.withColumn("boxes", detect_udf(F.col("frame_path")))
        .withColumn("model_version", F.lit(model_version))
        .withColumn("inferred_at", F.current_timestamp())
        .withColumn("alias_used", F.lit(alias))
        .select(
            "frame_id",
            "video_id",
            "boxes",
            "model_version",
            "inferred_at",
            "alias_used",
        )
    )


def merge_detections(detections: DataFrame) -> None:
    """Idempotently persist detections to the Delta detections table.

    Uses ``DeltaTable.merge`` keyed on ``frame_id`` so re-runs neither duplicate
    nor blindly append rows. When the table does not yet exist it is created
    once via a controlled ``saveAsTable`` (never a bare append).

    Parameters
    ----------
    detections:
        DataFrame produced by :func:`build_detections`.
    """
    if not spark.catalog.tableExists(detections_table):
        # First run: materialize the table with the canonical schema/columns.
        (
            detections.write.format("delta")
            .mode("overwrite")
            .saveAsTable(detections_table)
        )
        print(f"Created detections table '{detections_table}'.")
        return

    target = DeltaTable.forName(spark, detections_table)
    (
        target.alias("t")
        .merge(detections.alias("s"), "t.frame_id = s.frame_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print(f"Merged detections into '{detections_table}' (key: frame_id).")


def main() -> None:
    """Entry point: read unprocessed frames, score them, MERGE into detections."""
    frames: DataFrame = read_unprocessed_frames()

    pending_count: int = frames.count()
    if pending_count == 0:
        print("No unprocessed frames found — detections table is up to date.")
        return

    print(f"Scoring {pending_count} unprocessed frame(s) with alias '{alias}'.")

    detections: DataFrame = build_detections(frames)
    merge_detections(detections)

    print("Batch inference complete.")


# COMMAND ----------

main()
