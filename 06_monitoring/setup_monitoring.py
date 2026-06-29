# Databricks notebook source
"""Pipeline stage 06 - Monitoring: set up Lakehouse Monitoring on model detections.

This standalone Databricks notebook/script is the monitoring stage of the
volei-tactical YOLOv8 MLOps pipeline. It consumes the canonical
`{catalog}.{schema}.detections` table produced by the inference stage
(05_inference/batch_inference.py) and wires up Databricks Lakehouse Monitoring
(Unity Catalog quality monitors) to track inference quality drift over time.

Because `detections` stores nested predictions (`boxes` = array<struct<...>>),
Lakehouse Monitoring cannot profile it directly. This script therefore:

  1. MATERIALIZES a flattened, per-frame metrics table
     `{catalog}.{schema}.detections_metrics` with one row per frame:
         frame_id, video_id, inferred_at, n_detections, avg_conf
  2. Builds a baseline table `{catalog}.{schema}.detections_baseline` from the
     FIRST WEEK of inference data (a stable reference distribution).
  3. Creates a TimeSeries quality monitor on `detections_metrics`, with the
     baseline table attached so drift is computed against the reference window.

Drift alerts focus on the two quality signals that matter for this pipeline:
  - avg_conf       : mean detection confidence per frame.
  - n_detections   : number of detected volleyball players per frame.

Pipeline position: runs AFTER 05_inference (which writes `detections`) and is
typically scheduled to refresh on a cadence (see the scheduling note at the
bottom of this file).

Conventions: Unity Catalog three-part namespaces, Spark-first batch processing,
idempotent writes (controlled overwrite), Databricks Runtime ML 15.x.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# COMMAND ----------

# -----------------------------------------------------------------------------
# Widgets / configuration
# -----------------------------------------------------------------------------
# ZERO hardcoded namespaces: catalog and schema come from Databricks widgets so
# this notebook is reusable across environments (dev / staging / prod).
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")  # CONFIGURE
dbutils.widgets.text("schema", "volei_tactical", "Unity Catalog schema")  # CONFIGURE

CATALOG: str = dbutils.widgets.get("catalog")
SCHEMA: str = dbutils.widgets.get("schema")

# Length of the baseline window, in days, taken from the first observed
# inference timestamp. Seven days = one full week of reference behaviour.
BASELINE_WINDOW_DAYS: int = 7  # CONFIGURE

# Monitoring time granularity for the TimeSeries profile.
MONITOR_GRANULARITY: str = "1 day"  # CONFIGURE

# Canonical three-part names (do NOT change: other stages depend on these).
DETECTIONS_TABLE: str = f"{CATALOG}.{SCHEMA}.detections"
METRICS_TABLE: str = f"{CATALOG}.{SCHEMA}.detections_metrics"
BASELINE_TABLE: str = f"{CATALOG}.{SCHEMA}.detections_baseline"
OUTPUT_SCHEMA: str = f"{CATALOG}.{SCHEMA}"

spark: SparkSession = SparkSession.builder.getOrCreate()

# COMMAND ----------


def build_detections_metrics(spark: SparkSession, detections_table: str) -> DataFrame:
    """Flatten the nested ``detections`` table into per-frame quality metrics.

    The canonical ``detections`` table carries a nested ``boxes`` column of type
    ``array<struct<x1:float,y1:float,x2:float,y2:float,conf:float,cls:int>>``.
    Lakehouse Monitoring profiles scalar columns only, so we derive two scalar
    quality signals per frame:

      * ``n_detections`` - number of detected boxes (``size(boxes)``).
      * ``avg_conf``     - mean of ``boxes.conf`` for the frame (null/0 boxes -> 0.0).

    Args:
        spark: Active Spark session.
        detections_table: Three-part name of the source detections table.

    Returns:
        A Spark DataFrame with columns:
        ``frame_id``, ``video_id``, ``inferred_at``, ``n_detections``, ``avg_conf``.
    """
    detections = spark.table(detections_table)

    # Number of detections per frame; treat a null/empty array as zero.
    n_detections = F.coalesce(F.size(F.col("boxes")), F.lit(0)).cast("int")

    # Mean confidence across the per-frame boxes. `transform` projects the
    # struct array down to its `conf` values; `aggregate` then averages them.
    # Frames with no boxes get avg_conf = 0.0 (no detections -> no confidence).
    conf_array = F.transform(F.col("boxes"), lambda b: b["conf"].cast("double"))
    conf_sum = F.aggregate(conf_array, F.lit(0.0), lambda acc, c: acc + c)
    avg_conf = (
        F.when(n_detections > 0, conf_sum / n_detections)
        .otherwise(F.lit(0.0))
        .cast("double")
    )

    return (
        detections.select(
            F.col("frame_id"),
            F.col("video_id"),
            F.col("inferred_at"),
            n_detections.alias("n_detections"),
            avg_conf.alias("avg_conf"),
        )
    )


def write_metrics_table(metrics_df: DataFrame, metrics_table: str) -> None:
    """Materialize the flattened metrics DataFrame to a Delta table.

    Uses a controlled full overwrite (idempotent re-materialization of the
    derived view) rather than a bare append, so repeated runs converge to a
    single, consistent metrics snapshot. The table is created with Change Data
    Feed enabled, which Lakehouse Monitoring uses for incremental refreshes.

    Args:
        metrics_df: Per-frame metrics produced by :func:`build_detections_metrics`.
        metrics_table: Three-part name of the target metrics table.
    """
    (
        metrics_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .saveAsTable(metrics_table)
    )


def write_baseline_table(
    spark: SparkSession,
    metrics_table: str,
    baseline_table: str,
    window_days: int,
) -> Optional[str]:
    """Build the baseline table from the first ``window_days`` of metrics.

    The baseline is the reference distribution against which Lakehouse
    Monitoring computes drift. We take every metrics row whose ``inferred_at``
    falls in ``[min(inferred_at), min(inferred_at) + window_days)`` - i.e. the
    first week of inference activity. Written with a controlled overwrite so the
    baseline is reproducible across re-runs.

    Args:
        spark: Active Spark session.
        metrics_table: Three-part name of the source metrics table.
        baseline_table: Three-part name of the target baseline table.
        window_days: Width of the baseline window, in days, from the first
            observed timestamp.

    Returns:
        The ISO timestamp string of the baseline cutoff, or ``None`` if the
        metrics table has no rows yet.
    """
    metrics = spark.table(metrics_table)

    # Earliest inference timestamp anchors the baseline window.
    min_row = metrics.select(F.min("inferred_at").alias("min_ts")).first()
    if min_row is None or min_row["min_ts"] is None:
        # No inference data yet -> nothing to baseline against.
        return None

    min_ts = min_row["min_ts"]
    cutoff = min_ts + timedelta(days=window_days)

    baseline_df = metrics.filter(F.col("inferred_at") < F.lit(cutoff))

    (
        baseline_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(baseline_table)
    )

    return cutoff.isoformat()


def create_or_update_monitor(
    w: WorkspaceClient,
    metrics_table: str,
    baseline_table: str,
    output_schema: str,
    timestamp_col: str = "inferred_at",
    granularity: str = MONITOR_GRANULARITY,
) -> None:
    """Create (or recreate) the Lakehouse Monitor on the metrics table.

    Configures a TimeSeries monitor that profiles ``metrics_table`` per
    ``granularity`` window, anchored on ``timestamp_col``, and attaches
    ``baseline_table`` so drift metrics are computed against the first-week
    reference distribution. Drift on ``avg_conf`` (mean detection confidence)
    and ``n_detections`` (detections per frame) is the primary signal to alert
    on once profiling output exists.

    This call is idempotent at the workflow level: if a monitor already exists
    for the table, the existing one is left in place (re-creation would raise),
    so re-running this stage is safe.

    Args:
        w: Authenticated Databricks ``WorkspaceClient``.
        metrics_table: Three-part name of the table to monitor.
        baseline_table: Three-part name of the baseline/reference table.
        output_schema: Two-part ``catalog.schema`` where monitor output tables
            (profile + drift metrics) are written.
        timestamp_col: Event-time column driving the TimeSeries windows.
        granularity: Aggregation granularity for the TimeSeries profile.
    """
    try:
        w.quality_monitors.get(table_name=metrics_table)
        print(f"[monitoring] Monitor already exists for {metrics_table}; skipping create.")
        return
    except Exception:
        # No monitor yet -> proceed to create one below.
        pass

    w.quality_monitors.create(
        table_name=metrics_table,
        time_series=MonitorTimeSeries(
            timestamp_col=timestamp_col,
            granularities=[granularity],
        ),
        baseline_table_name=baseline_table,
        output_schema_name=output_schema,
    )
    print(
        f"[monitoring] Created TimeSeries monitor on {metrics_table} "
        f"(baseline={baseline_table}, output_schema={output_schema}).\n"
        "[monitoring] Drift to watch: avg_conf (mean confidence) and "
        "n_detections (detections per frame)."
    )


# COMMAND ----------

# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

# 1) Materialize the flattened per-frame metrics table from nested detections.
metrics_df = build_detections_metrics(spark, DETECTIONS_TABLE)
write_metrics_table(metrics_df, METRICS_TABLE)
print(f"[monitoring] Materialized {METRICS_TABLE} from {DETECTIONS_TABLE}.")

# 2) Build the first-week baseline used as the drift reference distribution.
cutoff = write_baseline_table(spark, METRICS_TABLE, BASELINE_TABLE, BASELINE_WINDOW_DAYS)
if cutoff is None:
    print(
        f"[monitoring] WARNING: {METRICS_TABLE} is empty; baseline {BASELINE_TABLE} "
        "was not populated. Run inference first, then re-run this stage."
    )
else:
    print(f"[monitoring] Built baseline {BASELINE_TABLE} (rows with inferred_at < {cutoff}).")

# 3) Create the Lakehouse Monitor (TimeSeries + baseline) on the metrics table.
w = WorkspaceClient()
create_or_update_monitor(
    w=w,
    metrics_table=METRICS_TABLE,
    baseline_table=BASELINE_TABLE,
    output_schema=OUTPUT_SCHEMA,
)

# COMMAND ----------

# -----------------------------------------------------------------------------
# Scheduling the monitor refresh via a Databricks Job
# -----------------------------------------------------------------------------
# Lakehouse Monitoring does NOT auto-refresh on every upstream write. To keep
# profile and drift metrics current, schedule a periodic refresh. Two options:
#
# A) Built-in monitor schedule: pass a `schedule=MonitorCronSchedule(...)` to
#    `w.quality_monitors.create(...)` so Databricks runs the refresh on a cron.
#
# B) Databricks Job (recommended for this pipeline, so the re-materialization of
#    `detections_metrics` and `detections_baseline` stays in sync with the
#    monitor). Create a scheduled Job whose task re-runs this notebook, OR whose
#    task triggers a refresh run directly. To trigger a refresh programmatically:
#
#        from databricks.sdk import WorkspaceClient
#        w = WorkspaceClient()
#        run = w.quality_monitors.run_refresh(
#            table_name=f"{CATALOG}.{SCHEMA}.detections_metrics"
#        )
#        # Optionally poll status with:
#        # w.quality_monitors.get_refresh(
#        #     table_name=f"{CATALOG}.{SCHEMA}.detections_metrics",
#        #     refresh_id=run.refresh_id,
#        # )
#
#    Wire that into a Databricks Job with a daily cron (e.g. quartz
#    "0 0 6 * * ?" for 06:00 daily), passing the same `catalog` / `schema`
#    job parameters used by the widgets above. Aligning the Job cron with the
#    "1 day" monitor granularity keeps one refreshed drift point per day.
# -----------------------------------------------------------------------------
