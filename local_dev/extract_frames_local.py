"""Local frame extraction — the portable core of 01_ingest/ingest_frames.py.

This mirrors the Databricks ingest step (same `frame_id` convention and the same
per-frame metadata schema) but writes to the LOCAL filesystem and a Parquet
manifest instead of a Unity Catalog Delta table. It lets us validate the OpenCV
sampling logic before running anything on a cluster.

Canonical contract kept identical to the Databricks step:
    frame_id      = f"{video_id}__{frame_number:06d}"
    manifest cols = video_id, frame_id, frame_path, timestamp_ms, width, height

Usage:
    python extract_frames_local.py --video <path.mp4> --interval 30 --max-frames 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import pandas as pd


def extract_frames(
    video_path: str,
    out_dir: str,
    frame_interval: int,
    max_frames: int | None = None,
    jpeg_quality: int = 92,
) -> pd.DataFrame:
    """Extract one frame every ``frame_interval`` frames from a single video.

    Args:
        video_path: Path to the source video file.
        out_dir: Directory where sampled ``.jpg`` frames are written
            (under ``out_dir/{video_id}/``).
        frame_interval: Sampling stride — keep one frame every N decoded frames.
        max_frames: Optional cap on how many frames to keep (handy for quick tests).
        jpeg_quality: JPEG encode quality (0-100).

    Returns:
        A pandas DataFrame with the canonical frame-manifest columns
        ``[video_id, frame_id, frame_path, timestamp_ms, width, height]``.
    """
    video_id = Path(video_path).stem
    video_out_dir = os.path.join(out_dir, video_id)
    os.makedirs(video_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    records: List[dict] = []
    frame_number = 0
    kept = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_number % frame_interval == 0:
                # Same frame_id convention as the Databricks ingest step.
                frame_id = f"{video_id}__{frame_number:06d}"
                frame_path = os.path.join(video_out_dir, f"{frame_id}.jpg")
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                height, width = frame.shape[:2]

                if not cv2.imwrite(frame_path, frame, encode_params):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")

                records.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "frame_path": frame_path,
                        "timestamp_ms": timestamp_ms,
                        "width": int(width),
                        "height": int(height),
                    }
                )
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break

            frame_number += 1
    finally:
        cap.release()

    return pd.DataFrame.from_records(
        records,
        columns=["video_id", "frame_id", "frame_path", "timestamp_ms", "width", "height"],
    )


def main() -> None:
    """CLI entry point: extract frames and write a Parquet + CSV manifest."""
    parser = argparse.ArgumentParser(description="Local frame extraction (ingest core).")
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "_work", "frames"),
        help="Output directory for extracted frames.",
    )
    parser.add_argument("--interval", type=int, default=30, help="Sampling stride (frames).")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional cap on frames kept."
    )
    args = parser.parse_args()

    manifest = extract_frames(
        video_path=args.video,
        out_dir=args.out_dir,
        frame_interval=args.interval,
        max_frames=args.max_frames,
    )

    work_dir = os.path.dirname(os.path.abspath(args.out_dir))
    parquet_path = os.path.join(work_dir, "frames.parquet")
    csv_path = os.path.join(work_dir, "frames.csv")
    manifest.to_parquet(parquet_path, index=False)
    manifest.to_csv(csv_path, index=False)

    print(f"Extracted {len(manifest)} frames -> {args.out_dir}")
    print(f"Manifest: {parquet_path}")
    print(manifest.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
