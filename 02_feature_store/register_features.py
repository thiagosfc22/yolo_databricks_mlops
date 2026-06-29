# Databricks notebook source
"""Register the raw frame feature table in the Databricks Feature Engineering store.

Pipeline position
------------------
Step 02 (feature_store). Runs AFTER 01_ingest has populated
``{catalog}.{schema}.frames`` and BEFORE 03_train consumes features.

What this file does
-------------------
1. Reads the canonical frames table ``{catalog}.{schema}.frames`` and selects the
   raw, model-independent columns needed downstream
   (``frame_id``, ``frame_path``, ``video_id``, ``timestamp_ms``).
2. Materializes those columns as the feature table
   ``{catalog}.{schema}.frame_features`` (primary key ``frame_id``) using the
   modern ``databricks.feature_engineering.FeatureEngineeringClient``.
   - First run: ``fe.create_table(...)`` with a description.
   - Subsequent runs: ``fe.write_table(mode="merge")`` for idempotent upserts
     keyed on ``frame_id`` (never a bare append).
3. Documents the table with a description (via ``create_table``) and Unity Catalog
   tags (via ``ALTER TABLE ... SET TAGS``).

Domain: volleyball position analysis (YOLOv8 detections on an 18x9 m court).

NOTE on model-derived features
-------------------------------
This step intentionally registers ONLY raw, model-independent features. Features
that depend on the model's predictions — e.g. ``detection_count`` (number of
detected players per frame) and ``mean_confidence`` (mean detection confidence
per frame) — are computed AFTER batch inference (step 05) writes
``{catalog}.{schema}.detections`` and are merged into ``frame_features`` later in
the pipeline. They are deliberately NOT created here to avoid a circular
dependency between feature engineering and inference.
"""

from typing import List

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# COMMAND ----------

# Widgets supply the Unity Catalog namespace. Three-part naming
# (catalog.schema.object) is mandatory for every UC object.
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")  # CONFIGURE
dbutils.widgets.text("schema", "volei_tactical", "Unity Catalog schema")  # CONFIGURE

CATALOG: str = dbutils.widgets.get("catalog")
SCHEMA: str = dbutils.widgets.get("schema")

# Canonical, contract-defined identifiers. Do NOT rename — other pipeline
# stages depend on these exact names.
FRAMES_TABLE: str = f"{CATALOG}.{SCHEMA}.frames"
FEATURE_TABLE: str = f"{CATALOG}.{SCHEMA}.frame_features"
PRIMARY_KEY: str = "frame_id"

# Unity Catalog tags applied to the feature table for discoverability/governance.
FEATURE_TABLE_TAGS: dict = {"domain": "volleyball", "stage": "raw_features"}

FEATURE_TABLE_DESCRIPTION: str = (
    "Raw, model-independent frame features for the volleyball position-analysis "
    "pipeline (volei-tactical). One row per extracted video frame, keyed by "
    "frame_id. Model-derived features (detection_count, mean_confidence) are "
    "merged in after batch inference."
)

# COMMAND ----------

spark: SparkSession = SparkSession.builder.getOrCreate()
fe: FeatureEngineeringClient = FeatureEngineeringClient()

# COMMAND ----------


def build_raw_features(frames_table: str) -> DataFrame:
    """Select the raw, model-independent feature columns from the frames table.

    Reads the canonical frames table and projects only the columns that form the
    raw feature set. ``frame_id`` is the primary key; duplicate rows for the same
    key are collapsed so the resulting DataFrame is unique per ``frame_id`` (a
    prerequisite for a clean feature-store MERGE).

    Args:
        frames_table: Fully qualified ``catalog.schema.frames`` table name.

    Returns:
        A Spark DataFrame with columns ``frame_id``, ``frame_path``, ``video_id``
        and ``timestamp_ms``, deduplicated on ``frame_id``.
    """
    frames: DataFrame = spark.table(frames_table)
    return (
        frames.select(
            F.col("frame_id"),
            F.col("frame_path"),
            F.col("video_id"),
            F.col("timestamp_ms"),
        )
        .dropDuplicates([PRIMARY_KEY])
    )


def table_exists(table_name: str) -> bool:
    """Return whether a Unity Catalog table already exists.

    Args:
        table_name: Fully qualified ``catalog.schema.table`` name.

    Returns:
        ``True`` if the table is registered in the catalog, else ``False``.
    """
    return spark.catalog.tableExists(table_name)


def apply_table_tags(table_name: str, tags: dict) -> None:
    """Attach Unity Catalog tags to a table via ``ALTER TABLE ... SET TAGS``.

    Tags are set idempotently: re-running overwrites the same keys with the same
    values, so this is safe to call on every pipeline run.

    Args:
        table_name: Fully qualified ``catalog.schema.table`` name.
        tags: Mapping of tag keys to tag values to set on the table.
    """
    set_clause: str = ", ".join(
        # Escape single quotes to keep the SET TAGS clause well-formed.
        f"'{key}' = '{value.replace(chr(39), chr(39) * 2)}'"
        for key, value in tags.items()
    )
    spark.sql(f"ALTER TABLE {table_name} SET TAGS ({set_clause})")


def register_feature_table(features_df: DataFrame) -> None:
    """Create or idempotently upsert the ``frame_features`` feature table.

    On the first run the table does not exist, so it is created with
    ``fe.create_table`` (which establishes the primary key and the description).
    On subsequent runs the table already exists, so rows are MERGED in with
    ``fe.write_table(mode="merge")`` keyed on ``frame_id`` — never a bare append,
    preserving idempotency.

    Args:
        features_df: Deduplicated raw-feature DataFrame keyed on ``frame_id``.
    """
    if table_exists(FEATURE_TABLE):
        # Idempotent upsert on the primary key for re-runs / incremental loads.
        fe.write_table(
            name=FEATURE_TABLE,
            df=features_df,
            mode="merge",
        )
        print(f"Merged {features_df.count()} rows into existing {FEATURE_TABLE}.")
    else:
        # First-time creation: declare the primary key and seed the data.
        fe.create_table(
            name=FEATURE_TABLE,
            primary_keys=[PRIMARY_KEY],
            df=features_df,
            description=FEATURE_TABLE_DESCRIPTION,
        )
        print(f"Created feature table {FEATURE_TABLE}.")


# COMMAND ----------


def main() -> None:
    """Entry point: build raw features, register the table, apply documentation tags."""
    feature_columns: List[str] = [
        "frame_id",
        "frame_path",
        "video_id",
        "timestamp_ms",
    ]
    print(f"Reading {FRAMES_TABLE}; projecting raw feature columns: {feature_columns}")

    features_df: DataFrame = build_raw_features(FRAMES_TABLE)
    register_feature_table(features_df)

    # Apply governance/discoverability tags after the table is guaranteed to exist.
    apply_table_tags(FEATURE_TABLE, FEATURE_TABLE_TAGS)
    print(f"Applied tags {FEATURE_TABLE_TAGS} to {FEATURE_TABLE}.")


# COMMAND ----------

main()
