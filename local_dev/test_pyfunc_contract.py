"""Local contract test for 03_train/yolo_pyfunc.py::YOLOPyfunc.

Loads the REAL pyfunc class (no copy), feeds it the frame manifest produced by
extract_frames_local.py, and asserts the inference contract holds:

    predict(DataFrame[path, frame_id]) -> DataFrame[frame_id, boxes, confs, classes]
    boxes[i] = {x1, y1, x2, y2, conf, cls}

It also renders one annotated frame to ``_work/annotated/`` so we have a visual
of YOLO detecting volleyball players locally (using the generic COCO ``person``
class until a custom model is trained).

Usage:
    python test_pyfunc_contract.py --weights <yolov8n.pt> [--manifest <frames.parquet>]
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
PYFUNC_FILE = HERE.parent / "03_train" / "yolo_pyfunc.py"
DEFAULT_MANIFEST = HERE / "_work" / "frames.parquet"
DEFAULT_WEIGHTS = HERE.parent.parent / "volei-tactical" / "yolov8n.pt"


class _Context:
    """Minimal stand-in for mlflow's PythonModelContext (just ``.artifacts``)."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        self.artifacts = artifacts


def _load_yolo_pyfunc_class() -> type:
    """Import YOLOPyfunc from the pipeline file (its dir name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("yolo_pyfunc", str(PYFUNC_FILE))
    assert spec and spec.loader, f"Cannot load module spec from {PYFUNC_FILE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YOLOPyfunc


def _annotate(frame_path: str, boxes: list[dict[str, Any]], out_path: str) -> None:
    """Draw detection boxes on a frame and save it (for the post visual)."""
    image = cv2.imread(frame_path)
    if image is None:
        return
    for b in boxes:
        p1 = (int(b["x1"]), int(b["y1"]))
        p2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(image, p1, p2, (0, 220, 0), 2)
        cv2.putText(
            image, f"{b['conf']:.2f}", (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, image)


def main() -> None:
    """Run the pyfunc over the manifest and assert the output contract."""
    parser = argparse.ArgumentParser(description="Local YOLOPyfunc contract test.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    assert os.path.exists(args.weights), f"weights not found: {args.weights}"
    assert os.path.exists(args.manifest), f"manifest not found: {args.manifest}"

    YOLOPyfunc = _load_yolo_pyfunc_class()
    model = YOLOPyfunc()
    model.load_context(_Context({"weights": args.weights}))

    manifest = pd.read_parquet(args.manifest)
    model_input = manifest[["frame_path", "frame_id"]].rename(columns={"frame_path": "path"})

    out = model.predict(None, model_input)

    # --- Assert the contract ---
    expected_cols = ["frame_id", "boxes", "confs", "classes"]
    assert list(out.columns) == expected_cols, f"columns mismatch: {list(out.columns)}"
    assert len(out) == len(model_input), "row count must be preserved"
    if out["boxes"].map(len).sum() > 0:
        sample = next(b for row in out["boxes"] for b in row)
        assert set(sample.keys()) == {"x1", "y1", "x2", "y2", "conf", "cls"}, sample.keys()
        assert isinstance(sample["cls"], int) and isinstance(sample["conf"], float)

    print("CONTRACT OK — columns:", list(out.columns))
    summary = out.assign(n_detections=out["boxes"].map(len))[["frame_id", "n_detections"]]
    print(summary.to_string(index=False))
    print(f"\nTotal detections across {len(out)} frames: {int(summary['n_detections'].sum())}")

    # Render the frame with the most detections, for the post.
    best_idx = out["boxes"].map(len).idxmax()
    best_frame_id = out.loc[best_idx, "frame_id"]
    frame_path = manifest.loc[manifest["frame_id"] == best_frame_id, "frame_path"].iloc[0]
    annotated_path = str(HERE / "_work" / "annotated" / f"{best_frame_id}__det.jpg")
    _annotate(frame_path, out.loc[best_idx, "boxes"], annotated_path)
    print(f"\nAnnotated sample ({len(out.loc[best_idx, 'boxes'])} dets): {annotated_path}")


if __name__ == "__main__":
    main()
