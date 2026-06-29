# local_dev — prove the pipeline on a laptop, before Databricks

These scripts validate the **portable core** of the pipeline locally (no workspace),
so that going to Databricks is about infrastructure, not logic. See
[`../docs/WALKTHROUGH.md`](../docs/WALKTHROUGH.md) for the narrated step-by-step.

## Setup

```bash
conda env create -f ../environment.yml   # includes openjdk for local PySpark
conda activate yolo-databricks
```

## Run order

| # | Script | Proves | Mirrors |
|---|---|---|---|
| 1 | `extract_frames_local.py` | OpenCV frame sampling + `frame_id` convention + manifest schema | `01_ingest/ingest_frames.py` |
| 2 | `test_pyfunc_contract.py` | `YOLOPyfunc.predict` returns `[frame_id, boxes, confs, classes]`; renders an annotated frame | `03_train/yolo_pyfunc.py` |
| 3 | `test_mlflow_packaging.py` | `log_model` → `load_model` → `predict` round-trips (local sqlite tracking) | `03_train` + `05_inference` |
| 4 | `test_spark_udf_delta.py` | Spark `pandas_udf` + Delta `MERGE`, idempotent; per-worker model load | `05_inference/batch_inference.py` |

```bash
cd local_dev
python extract_frames_local.py \
  --video ../../posts/02-volei-brasil-italia/media/brasil-italia-nDeXUywqnIs-clip.mp4 \
  --interval 30 --max-frames 12
python test_pyfunc_contract.py
python test_mlflow_packaging.py
python test_spark_udf_delta.py
```

All working artifacts land under `local_dev/_work/` (git-ignored): extracted frames,
the `frames.parquet` manifest, a local MLflow sqlite store, the annotated sample, and
a local Delta `detections` table.

> These tests reuse `../../volei-tactical/yolov8n.pt` (generic COCO weights) and the
> Brazil×Italy VNL clip. The COCO model detects *every* person (players, bench, crowd);
> training a custom court-filtered model is exactly stage `03`'s job.
