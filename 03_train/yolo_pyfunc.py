"""YOLO pyfunc wrapper for the volei-tactical MLOps pipeline.

Pipeline position
-----------------
Stage 03 (train / package). This module defines :class:`YOLOPyfunc`, the
``mlflow.pyfunc.PythonModel`` that wraps an Ultralytics YOLOv8 detector so the
trained weights can be logged once and then loaded uniformly across the
pipeline (registry promotion in stage 04, batch inference in stage 05).

The class is intentionally framework-agnostic at the MLflow boundary: it takes
a pandas ``DataFrame`` of image paths and returns a pandas ``DataFrame`` of
detections. This keeps the inference contract stable regardless of how the
model is served (Spark ``pandas_udf``, batch job, or model serving endpoint).

Inference contract (MUST stay in sync with 05_inference/batch_inference.py)
---------------------------------------------------------------------------
- ``predict`` receives a ``pandas.DataFrame`` with a ``"path"`` column (string)
  and optionally a ``"frame_id"`` column (string).
- ``predict`` returns a ``pandas.DataFrame`` with EXACT columns:
      ``frame_id``  (str)
      ``boxes``     (list per frame of dicts {x1, y1, x2, y2, conf, cls})
      ``confs``     (list of float)
      ``classes``   (list of int)
- ``boxes`` dict field types: x1/y1/x2/y2/conf -> float, cls -> int. This maps
  to the Spark return type used by the inference UDF:
      array<struct<x1:float,y1:float,x2:float,y2:float,conf:float,cls:int>>
- The model is logged with a "weights" artifact pointing at ``best.pt`` and is
  loaded elsewhere via:
      mlflow.pyfunc.load_model(f"models:/{catalog}.{schema}.{model_name}@{alias}")

Robustness
----------
Each row is processed inside its own try/except. A corrupt frame, an
unreadable path, or a per-image inference error yields empty results
(``boxes=[]``, ``confs=[]``, ``classes=[]``) for that row WITHOUT breaking the
rest of the batch. This is required so a single bad frame never fails an entire
Spark inference task.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import mlflow
import pandas as pd

# --------------------------------------------------------------------------- #
# CONFIGURE: inference hyper-parameters baked into the packaged model.
# These are deliberately conservative defaults for volleyball-player detection.
# Override here before logging the model if the dataset/runtime requires it.
# --------------------------------------------------------------------------- #
PREDICT_CONF_THRESHOLD: float = 0.25  # CONFIGURE: minimum confidence to keep a detection
PREDICT_IOU_THRESHOLD: float = 0.45  # CONFIGURE: NMS IoU threshold
PREDICT_IMG_SIZE: int = 640  # CONFIGURE: inference image size (square), DRML 15.x friendly
PREDICT_MAX_DETECTIONS: int = 300  # CONFIGURE: max detections kept per frame
PREDICT_DEVICE: str = "cpu"  # CONFIGURE: "cpu" or "cuda:0"; CPU keeps batch UDFs portable

# Exact output column order required by the inference contract. Do not reorder.
OUTPUT_COLUMNS: List[str] = ["frame_id", "boxes", "confs", "classes"]


class YOLOPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around an Ultralytics YOLOv8 detector.

    The wrapped model is loaded from the ``"weights"`` artifact (a ``best.pt``
    checkpoint) at serving time. Prediction maps a frame of image paths to
    bounding-box detections following the pipeline-wide inference contract.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the YOLO model from the logged ``best.pt`` weights artifact.

        Ultralytics and OpenCV are imported lazily here so the dependencies are
        only required inside the serving/inference environment, not at the time
        the class is defined or the module is imported.

        Args:
            context: MLflow context exposing logged artifacts. The detector
                weights are expected under the ``"weights"`` artifact key
                (pointing at ``best.pt``).
        """
        # Ultralytics writes a ``settings.json`` under its user-config dir on
        # first import. On serving/UDF executors the default location can be
        # unwritable for the run uid, raising a ``PermissionError``. Pin it to a
        # fresh writable temp dir BEFORE importing ultralytics so the packaged
        # model loads on any executor (model serving, batch ``pandas_udf``).
        import tempfile

        if "YOLO_CONFIG_DIR" not in os.environ:
            os.environ["YOLO_CONFIG_DIR"] = tempfile.mkdtemp(prefix="ultralytics_")

        # Lazy imports: keep heavy CV deps out of the module import path.
        from ultralytics import YOLO  # noqa: WPS433 (intentional lazy import)

        weights_path: str = context.artifacts["weights"]
        self.model = YOLO(weights_path)

    def _predict_one(self, path: str) -> Dict[str, List[Any]]:
        """Run detection on a single image path.

        Args:
            path: Absolute path to a single frame image (e.g. a PNG/JPG file,
                typically under ``/Volumes/...``).

        Returns:
            A dict with keys ``boxes`` (list of detection dicts), ``confs``
            (list of float), and ``classes`` (list of int) for this frame.
        """
        # Lazy import: only needed during actual inference.
        import cv2  # noqa: WPS433 (intentional lazy import)

        # Read the image with OpenCV. A missing/corrupt file yields None, which
        # we treat as an empty detection set (raised below, caught by caller).
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read image at path: {path}")

        # Convert BGR (OpenCV) -> RGB (what Ultralytics expects for arrays).
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=image_rgb,
            conf=PREDICT_CONF_THRESHOLD,
            iou=PREDICT_IOU_THRESHOLD,
            imgsz=PREDICT_IMG_SIZE,
            max_det=PREDICT_MAX_DETECTIONS,
            device=PREDICT_DEVICE,
            verbose=False,
        )

        boxes: List[Dict[str, Any]] = []
        confs: List[float] = []
        classes: List[int] = []

        # Ultralytics returns a list of Results (one per input image); we feed a
        # single image, so iterate defensively over whatever it returns.
        for result in results:
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            # xyxy: (N, 4) tensor of pixel corner coords; conf: (N,); cls: (N,).
            xyxy = result_boxes.xyxy.cpu().numpy()
            conf_arr = result_boxes.conf.cpu().numpy()
            cls_arr = result_boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, conf_arr, cls_arr):
                conf_f = float(conf)
                cls_i = int(cls)
                boxes.append(
                    {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "conf": conf_f,
                        "cls": cls_i,
                    }
                )
                confs.append(conf_f)
                classes.append(cls_i)

        return {"boxes": boxes, "confs": confs, "classes": classes}

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run batched detection over a frame of image paths.

        Args:
            context: MLflow context (unused at predict time; the model is
                already loaded in :meth:`load_context`).
            model_input: A pandas ``DataFrame`` with a required ``"path"``
                column (string) and an optional ``"frame_id"`` column (string).
                When ``"frame_id"`` is absent, it is derived from the file name
                (basename without extension), consistent with the
                ``f"{video_id}__{frame_number:06d}"`` convention.

        Returns:
            A pandas ``DataFrame`` with the EXACT columns
            ``["frame_id", "boxes", "confs", "classes"]`` (one row per input
            row, order preserved). Rows that fail to process yield empty
            detection lists rather than raising, so a single bad frame never
            breaks the batch.
        """
        has_frame_id = "frame_id" in model_input.columns

        records: List[Dict[str, Any]] = []
        # itertuples keeps row order and is faster than iterrows for wide frames.
        for row in model_input.itertuples(index=False):
            row_dict = row._asdict()
            path = str(row_dict["path"])

            # Resolve frame_id: explicit column wins, else basename w/o extension.
            if has_frame_id and row_dict.get("frame_id") is not None:
                frame_id = str(row_dict["frame_id"])
            else:
                frame_id = os.path.splitext(os.path.basename(path))[0]

            # Per-row isolation: any failure -> empty detections, keep going.
            try:
                detections = self._predict_one(path)
            except Exception:  # noqa: BLE001 (intentional: never break the batch)
                detections = {"boxes": [], "confs": [], "classes": []}

            records.append(
                {
                    "frame_id": frame_id,
                    "boxes": detections["boxes"],
                    "confs": detections["confs"],
                    "classes": detections["classes"],
                }
            )

        # Build the output with the exact, contract-mandated column order.
        return pd.DataFrame(records, columns=OUTPUT_COLUMNS)
