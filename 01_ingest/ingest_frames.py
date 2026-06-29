# Databricks notebook source
"""Frame ingestion job (pipeline step 01_ingest).

Position in the pipeline:
    This is the FIRST step of the volei-tactical MLOps pipeline. It reads raw
    volleyball match videos from a Unity Catalog volume, decodes them with
    OpenCV, samples one frame every ``frame_interval`` frames, persists each
    sampled frame as a ``.jpg`` into a UC volume, and registers per-frame
    metadata into the canonical Delta table ``{catalog}.{schema}.frames``.

What it does, concretely:
    1. Lists video files (.mp4/.mov/.avi) in
       ``/Volumes/{catalog}/{schema}/raw_videos/``.
    2. For each video, opens a ``cv2.VideoCapture`` and iterates over its
       frames. Every ``frame_interval`` frames it writes the frame as a JPEG to
       ``/Volumes/{catalog}/{schema}/dataset/frames/{video_id}/{frame_id}.jpg``
       and accumulates a metadata record. Extracted frames live under a
       ``frames/`` subdirectory of the documented ``dataset`` volume, kept
       separate from the Ultralytics ``images/``/``labels/`` training layout.
    3. Builds a Spark DataFrame with the EXACT ``frames`` schema and MERGEs it
       (idempotently) by ``frame_id`` into the Delta table, creating the table
       on first run if it does not exist.

Idempotency:
    Re-running the job for the same videos re-derives the same ``frame_id``
    values (``f"{video_id}__{frame_number:06d}"``) and MERGEs by that key, so
    reprocessing never produces duplicate rows and never does a bare append.

Assumptions (documented in the README):
    The catalog, the schema, the ``raw_videos`` volume and the ``dataset``
    volume already exist. This job does not create UC objects other than the
    ``frames`` Delta table.
"""

from __future__ import annotations

import os
from typing import Any

import cv2

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# --------------------------------------------------------------------------- #
# Top-of-file configuration.
# Everything tunable lives in dbutils.widgets; values below are only defaults
# and small constants. Items tagged "# CONFIGURE" are the ones an operator may
# legitimately want to change per deployment.
# --------------------------------------------------------------------------- #

# CONFIGURE: default Unity Catalog catalog used when the widget is left empty.
DEFAULT_CATALOG: str = "main"
# CONFIGURE: default Unity Catalog schema used when the widget is left empty.
DEFAULT_SCHEMA: str = "volei_tactical"
# CONFIGURE: default sampling stride (save one frame every N decoded frames).
DEFAULT_FRAME_INTERVAL: str = "30"
# CONFIGURE: JPEG quality (0-100) used when encoding sampled frames to disk.
JPEG_QUALITY: int = 95

# Video container extensions that this job knows how to decode.
SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".avi")

# EXACT canonical schema of the {catalog}.{schema}.frames table. Other pipeline
# steps depend on these column names and types, so do NOT change them here.
FRAMES_SCHEMA: StructType = StructType(
    [
        StructField("video_id", StringType(), nullable=False),
        StructField("frame_id", StringType(), nullable=False),
        StructField("frame_path", StringType(), nullable=False),
        StructField("timestamp_ms", LongType(), nullable=True),
        StructField("width", IntegerType(), nullable=True),
        StructField("height", IntegerType(), nullable=True),
        # extracted_at is added in Spark via current_timestamp(); it is not part
        # of the Python-side records, hence it is appended later, not here.
    ]
)


def get_dbutils(spark: SparkSession) -> Any:
    """Return the active ``dbutils`` handle.

    Works both inside a Databricks notebook (where ``dbutils`` is injected into
    the global namespace) and inside a job/`spark-submit` context (where it must
    be obtained from :class:`DBUtils`).

    Args:
        spark: The active :class:`~pyspark.sql.SparkSession`.

    Returns:
        The ``dbutils`` object exposing ``widgets``, ``fs`` and ``secrets``.
    """
    try:
        # Available automatically when running as a Databricks notebook.
        return dbutils  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        from pyspark.dbutils import DBUtils

        return DBUtils(spark)


def read_widgets(dbutils_handle: Any) -> tuple[str, str, int]:
    """Read and validate the job parameters from ``dbutils.widgets``.

    Three widgets are declared and read:
        * ``catalog`` (string) - Unity Catalog catalog name.
        * ``schema`` (string) - Unity Catalog schema name.
        * ``frame_interval`` (string) - sampling stride, converted to ``int``.

    Args:
        dbutils_handle: The ``dbutils`` object returned by :func:`get_dbutils`.

    Returns:
        A tuple ``(catalog, schema, frame_interval)`` with ``frame_interval``
        parsed as a positive ``int``.

    Raises:
        ValueError: If ``frame_interval`` is not a positive integer, or if the
            ``catalog``/``schema`` values are empty.
    """
    dbutils_handle.widgets.text("catalog", DEFAULT_CATALOG, "Unity Catalog catalog")
    dbutils_handle.widgets.text("schema", DEFAULT_SCHEMA, "Unity Catalog schema")
    dbutils_handle.widgets.text(
        "frame_interval", DEFAULT_FRAME_INTERVAL, "Frame sampling stride"
    )

    catalog: str = dbutils_handle.widgets.get("catalog").strip()
    schema: str = dbutils_handle.widgets.get("schema").strip()
    frame_interval_raw: str = dbutils_handle.widgets.get("frame_interval").strip()

    if not catalog or not schema:
        raise ValueError("Widgets 'catalog' and 'schema' must both be non-empty.")

    try:
        frame_interval: int = int(frame_interval_raw)
    except ValueError as exc:
        raise ValueError(
            f"Widget 'frame_interval' must be an integer, got: {frame_interval_raw!r}"
        ) from exc

    if frame_interval <= 0:
        raise ValueError(
            f"Widget 'frame_interval' must be a positive integer, got: {frame_interval}"
        )

    return catalog, schema, frame_interval


def list_video_files(raw_videos_dir: str) -> list[str]:
    """List the supported video files inside the raw-videos volume.

    Uses ``os.listdir`` on the local ``/Volumes`` FUSE path so that file names
    can be filtered by extension in pure Python.

    Args:
        raw_videos_dir: Absolute ``/Volumes`` path of the raw-videos volume,
            e.g. ``/Volumes/main/volei_tactical/raw_videos``.

    Returns:
        A sorted list of file names (not full paths) whose extension is in
        :data:`SUPPORTED_VIDEO_EXTENSIONS`.
    """
    if not os.path.isdir(raw_videos_dir):
        raise FileNotFoundError(f"Raw videos directory does not exist: {raw_videos_dir}")

    names: list[str] = [
        name
        for name in os.listdir(raw_videos_dir)
        if name.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS)
    ]
    return sorted(names)


def build_frame_id(video_id: str, frame_number: int) -> str:
    """Build the canonical ``frame_id`` for a given video and frame number.

    The convention (shared across the whole pipeline) is
    ``f"{video_id}__{frame_number:06d}"`` with TWO underscores between the two
    fields and the frame number zero-padded to six digits.

    Args:
        video_id: The video identifier (the file name without its extension).
        frame_number: The zero-based index of the frame within the video.

    Returns:
        The canonical ``frame_id`` string.
    """
    return f"{video_id}__{frame_number:06d}"


def extract_frames_from_video(
    video_path: str,
    video_id: str,
    frames_out_dir: str,
    frame_interval: int,
) -> list[dict[str, Any]]:
    """Decode one video and persist sampled frames as JPEG files.

    Iterates over every decoded frame; whenever the frame index is a multiple
    of ``frame_interval`` the frame is written to
    ``{frames_out_dir}/{video_id}/{frame_id}.jpg`` and a metadata record is
    accumulated.

    Args:
        video_path: Absolute ``/Volumes`` path to the source video file.
        video_id: Logical video identifier (file name without extension).
        frames_out_dir: Absolute ``/Volumes`` path of the extracted-frames
            directory (``frames/`` subdirectory of the ``dataset`` volume).
        frame_interval: Save one frame every ``frame_interval`` decoded frames.

    Returns:
        A list of metadata dictionaries, one per saved frame, each with keys
        matching the non-timestamp columns of :data:`FRAMES_SCHEMA`:
        ``video_id``, ``frame_id``, ``frame_path``, ``timestamp_ms``,
        ``width`` and ``height``.

    Raises:
        RuntimeError: If the video cannot be opened by OpenCV.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV failed to open video: {video_path}")

    # Per-video output directory inside the dataset volume's frames/ subdir.
    video_out_dir: str = os.path.join(frames_out_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    encode_params: list[int] = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    records: list[dict[str, Any]] = []
    frame_number: int = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                # No more frames (or a decode error): stop iterating.
                break

            if frame_number % frame_interval == 0:
                # Capture the presentation timestamp BEFORE writing the file;
                # CAP_PROP_POS_MSEC reflects the position of the frame just read.
                timestamp_ms: int = int(capture.get(cv2.CAP_PROP_POS_MSEC))

                frame_id: str = build_frame_id(video_id, frame_number)
                frame_path: str = os.path.join(video_out_dir, f"{frame_id}.jpg")

                # frame.shape is (height, width, channels) for a BGR image.
                height: int = int(frame.shape[0])
                width: int = int(frame.shape[1])

                if not cv2.imwrite(frame_path, frame, encode_params):
                    raise RuntimeError(f"Failed to write JPEG frame: {frame_path}")

                records.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "frame_path": frame_path,
                        "timestamp_ms": timestamp_ms,
                        "width": width,
                        "height": height,
                    }
                )

            frame_number += 1
    finally:
        # Always release the native handle, even if writing raised.
        capture.release()

    return records


def build_frames_dataframe(
    spark: SparkSession, records: list[dict[str, Any]]
) -> DataFrame:
    """Build the Spark DataFrame matching the canonical ``frames`` schema.

    The ``extracted_at`` column is added here via ``current_timestamp()`` so
    that ingestion time is recorded server-side at write time.

    Args:
        spark: The active :class:`~pyspark.sql.SparkSession`.
        records: Per-frame metadata dictionaries produced by
            :func:`extract_frames_from_video`.

    Returns:
        A Spark DataFrame with columns ``video_id``, ``frame_id``,
        ``frame_path``, ``timestamp_ms``, ``width``, ``height`` and
        ``extracted_at`` (in that canonical order).
    """
    frames_df: DataFrame = spark.createDataFrame(records, schema=FRAMES_SCHEMA)
    return frames_df.withColumn("extracted_at", F.current_timestamp())


def merge_frames(spark: SparkSession, frames_df: DataFrame, frames_table: str) -> None:
    """Idempotently MERGE the frames DataFrame into the Delta target table.

    Creates the Delta table on first run if it does not already exist, then
    MERGEs by the logical primary key ``frame_id``: matched rows are updated and
    new rows are inserted. This guarantees no duplicates and no bare appends
    when the job is reprocessed.

    Args:
        spark: The active :class:`~pyspark.sql.SparkSession`.
        frames_df: DataFrame produced by :func:`build_frames_dataframe`.
        frames_table: Fully qualified target table ``{catalog}.{schema}.frames``.
    """
    if not spark.catalog.tableExists(frames_table):
        # First run: create the managed Delta table with the exact schema by
        # writing the initial batch. Subsequent runs go through MERGE.
        frames_df.write.format("delta").saveAsTable(frames_table)
        return

    target: DeltaTable = DeltaTable.forName(spark, frames_table)
    (
        target.alias("t")
        .merge(frames_df.alias("s"), "t.frame_id = s.frame_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )


def main() -> None:
    """Entry point: ingest every raw video into the ``frames`` table.

    Reads the job widgets, resolves the canonical volume/table paths, decodes
    and samples every supported video, and MERGEs all sampled frames into
    ``{catalog}.{schema}.frames`` in a single idempotent write.
    """
    spark: SparkSession = SparkSession.builder.getOrCreate()
    dbutils_handle: Any = get_dbutils(spark)

    catalog, schema, frame_interval = read_widgets(dbutils_handle)

    # Canonical Unity Catalog paths (three-part namespace + /Volumes paths).
    # Extracted JPEGs go under the documented ``dataset`` volume, in a
    # dedicated ``frames/`` subdirectory (kept apart from the Ultralytics
    # ``images/``/``labels/`` training layout that also lives in that volume).
    frames_table: str = f"{catalog}.{schema}.frames"
    raw_videos_dir: str = f"/Volumes/{catalog}/{schema}/raw_videos"
    frames_out_dir: str = f"/Volumes/{catalog}/{schema}/dataset/frames"

    print(f"[ingest_frames] catalog={catalog} schema={schema} "
          f"frame_interval={frame_interval}")
    print(f"[ingest_frames] reading videos from: {raw_videos_dir}")
    print(f"[ingest_frames] writing frames to:   {frames_out_dir}")
    print(f"[ingest_frames] target table:        {frames_table}")

    video_files: list[str] = list_video_files(raw_videos_dir)
    if not video_files:
        print("[ingest_frames] No supported video files found; nothing to ingest.")
        return

    all_records: list[dict[str, Any]] = []
    for file_name in video_files:
        # video_id is the file name without its extension.
        video_id: str = os.path.splitext(file_name)[0]
        video_path: str = os.path.join(raw_videos_dir, file_name)

        print(f"[ingest_frames] processing video_id={video_id} ({file_name}) ...")
        records: list[dict[str, Any]] = extract_frames_from_video(
            video_path=video_path,
            video_id=video_id,
            frames_out_dir=frames_out_dir,
            frame_interval=frame_interval,
        )
        print(f"[ingest_frames]   -> sampled {len(records)} frames")
        all_records.extend(records)

    if not all_records:
        print("[ingest_frames] No frames sampled from any video; nothing to write.")
        return

    frames_df: DataFrame = build_frames_dataframe(spark, all_records)
    merge_frames(spark, frames_df, frames_table)

    print(f"[ingest_frames] MERGE complete: {len(all_records)} frames upserted into "
          f"{frames_table}")


# COMMAND ----------

if __name__ == "__main__":
    main()
