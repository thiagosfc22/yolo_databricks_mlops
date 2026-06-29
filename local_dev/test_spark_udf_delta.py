"""Local Spark `pandas_udf` + Delta MERGE test — mirrors 05_inference/batch_inference.py.

The real batch_inference.py is Databricks-bound (it reads `dbutils` widgets and
three-part Unity Catalog table names at import time), so it cannot be imported as-is
locally. This script reproduces its EXACT logic against a local Spark + Delta setup
and path-based Delta tables:

    1. load the pyfunc with mlflow, broadcast it to executors
    2. @pandas_udf(array<struct<...>>) wrapping model.predict -> the "boxes" column
    3. read frames, LEFT ANTI JOIN already-scored frames (incremental)
    4. attach provenance columns (model_version, inferred_at, alias_used)
    5. Delta MERGE by frame_id (create-on-first-run, never blind append)

It runs the whole thing TWICE to prove idempotency: the second pass finds zero
unprocessed frames. This is the most "Databricks-like" piece we can prove on a laptop;
on the cluster only the infra changes (UC names + /Volumes), not this logic.

Usage:
    python test_spark_udf_delta.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.pandas.functions import pandas_udf

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "03_train"
PYFUNC_FILE = TRAIN_DIR / "yolo_pyfunc.py"
WEIGHTS = HERE.parent.parent / "volei-tactical" / "yolov8n.pt"
MANIFEST = HERE / "_work" / "frames.parquet"
MLFLOW_DB = HERE / "_work" / "mlflow.db"
ARTIFACTS = HERE / "_work" / "mlartifacts"
DETECTIONS_PATH = str(HERE / "_work" / "delta" / "detections")

# Same binding contract string as batch_inference.py.
BOXES_RETURN_TYPE = "array<struct<x1:float,y1:float,x2:float,y2:float,conf:float,cls:int>>"
ALIAS = "champion"  # mirrors the default alias used on Databricks

sys.path.insert(0, str(TRAIN_DIR))
import yolo_pyfunc  # noqa: E402

# Driver-resolved local model dir + per-worker model cache (set in main()).
MODEL_LOCAL_PATH: str = ""
_WORKER_MODEL: dict = {}


def build_spark() -> SparkSession:
    """Local SparkSession with the Delta Lake extension enabled."""
    from delta import configure_spark_with_delta_pip

    builder = (
        SparkSession.builder.appName("local-yolo-inference")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def get_model_uri() -> tuple[str, str]:
    """Return (model_uri, model_version), logging a model first if none exists."""
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    exp_name = "local_smoke_yolo_cv"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=f"file:{ARTIFACTS}")
    mlflow.set_experiment(exp_name)

    runs = mlflow.search_runs(experiment_names=[exp_name], max_results=1)
    if len(runs) == 0:
        with mlflow.start_run() as run:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=yolo_pyfunc.YOLOPyfunc(),
                artifacts={"weights": str(WEIGHTS)},
                code_paths=[str(PYFUNC_FILE)],
                pip_requirements=["ultralytics>=8.2.0", "opencv-python-headless>=4.9.0"],
            )
            run_id = run.info.run_id
    else:
        run_id = runs.iloc[0]["run_id"]
    return f"runs:/{run_id}/model", run_id[:12]


def run_inference_pass(spark: SparkSession, model_version: str) -> int:
    """One incremental inference pass; returns the number of frames scored."""
    frames: DataFrame = (
        spark.read.parquet(str(MANIFEST)).select("frame_id", "video_id", "frame_path")
    )

    # LEFT ANTI JOIN against already-scored frames (incremental / idempotent).
    if os.path.exists(DETECTIONS_PATH):
        already = spark.read.format("delta").load(DETECTIONS_PATH).select("frame_id")
        frames = frames.join(already, on="frame_id", how="left_anti")

    pending = frames.count()
    if pending == 0:
        print("  -> no unprocessed frames; detections are up to date.")
        return 0

    detections = (
        frames.withColumn("boxes", detect_udf(F.col("frame_path")))
        .withColumn("model_version", F.lit(model_version))
        .withColumn("inferred_at", F.current_timestamp())
        .withColumn("alias_used", F.lit(ALIAS))
        .select("frame_id", "video_id", "boxes", "model_version", "inferred_at", "alias_used")
    )

    if not os.path.exists(DETECTIONS_PATH):
        detections.write.format("delta").mode("overwrite").save(DETECTIONS_PATH)
        print(f"  -> created detections Delta table, scored {pending} frame(s).")
    else:
        from delta.tables import DeltaTable

        target = DeltaTable.forPath(spark, DETECTIONS_PATH)
        (
            target.alias("t")
            .merge(detections.alias("s"), "t.frame_id = s.frame_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        print(f"  -> merged {pending} frame(s) by frame_id.")
    return pending


def main() -> None:
    """Run two incremental passes, then summarize the Delta detections table."""
    assert MANIFEST.exists(), "run extract_frames_local.py first"
    model_uri, model_version = get_model_uri()
    print(f"Model: {model_uri} (version tag: {model_version})")

    spark = build_spark()
    spark.sparkContext.setLogLevel("ERROR")

    # Resolve a LOCAL model dir on the driver, then load the model lazily PER WORKER
    # from that path. We do NOT broadcast a pre-loaded pyfunc: a worker can't unpickle
    # the custom YOLOPyfunc class without its code, whereas mlflow.pyfunc.load_model
    # ships the model's code_paths to whichever process loads it. This is the robust
    # pattern (see WALKTHROUGH Step 4 note; applies to batch_inference.py too).
    global detect_udf, MODEL_LOCAL_PATH
    MODEL_LOCAL_PATH = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    @pandas_udf(BOXES_RETURN_TYPE)
    def _detect_udf(paths: pd.Series) -> pd.Series:
        import mlflow as _mlflow

        if "model" not in _WORKER_MODEL:
            _WORKER_MODEL["model"] = _mlflow.pyfunc.load_model(MODEL_LOCAL_PATH)
        model = _WORKER_MODEL["model"]
        predictions = model.predict(pd.DataFrame({"path": paths.to_numpy()}))
        boxes = predictions["boxes"].reset_index(drop=True)
        boxes.index = paths.index
        return boxes

    detect_udf = _detect_udf

    print("\nPass 1 (cold):")
    run_inference_pass(spark, model_version)
    print("Pass 2 (idempotency check):")
    run_inference_pass(spark, model_version)

    # Summarize what landed in Delta.
    det = spark.read.format("delta").load(DETECTIONS_PATH)
    total_rows = det.count()
    summary = (
        det.select(
            "frame_id",
            F.size("boxes").alias("n_detections"),
            F.expr("aggregate(boxes, cast(0.0 as double), (acc, b) -> acc + b.conf)").alias("sum_conf"),
        )
        .withColumn("avg_conf", F.round(F.col("sum_conf") / F.col("n_detections"), 3))
        .drop("sum_conf")
        .orderBy("frame_id")
    )
    print(f"\nDelta detections table: {total_rows} rows (1 per frame)")
    summary.show(truncate=False)
    print(f"Delta path: {DETECTIONS_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
