# Walkthrough — running the YOLO + Databricks MLOps pipeline

This is the running log of how the pipeline is actually built, tested, and shipped —
first locally, then on Databricks. It doubles as the working runbook for the repo.

## The mental model: what runs where

The pipeline is **Databricks-first** (Unity Catalog, `/Volumes`, Spark, MLflow UC
registry). But the *valuable core* — frame extraction, YOLOv8 detection, and the
MLflow pyfunc contract — is plain Python and can be validated locally before any
cluster spins up. Knowing the split is what keeps the dev loop fast.

| Concern | Local (this machine) | Databricks (workspace) |
|---|---|---|
| Frame extraction (`cv2`) | ✅ on a local video | ✅ from `/Volumes/.../raw_videos` |
| YOLOv8 detection | ✅ `yolov8n.pt` on real frames | ✅ same code, GPU cluster |
| `YOLOPyfunc` predict contract | ✅ import & call directly | ✅ packaged in the model |
| MLflow log/load model | ✅ local file-based tracking | ✅ Unity Catalog registry |
| Spark `pandas_udf` + Delta MERGE | ✅ local Spark (Java in the conda env) | ✅ cluster Spark |
| Unity Catalog 3-part names, `/Volumes` | ❌ needs workspace | ✅ |
| Feature Engineering client | ❌ needs workspace | ✅ |
| Model Serving endpoint | ❌ needs workspace | ✅ (optional, billed) |
| Lakehouse Monitoring | ❌ needs workspace | ✅ |

The rule of thumb: **anything that touches `dbutils`, `/Volumes`, or a three-part
`catalog.schema.object` name is Databricks-only**; everything else we prove locally
first under `local_dev/`.

---

## Step 0 — the local dev environment

A single conda env carries the whole local toolchain, including a JDK (`openjdk`)
so local PySpark runs without any system Java.

```bash
conda env create -f environment.yml
conda activate yolo-databricks
```

What's inside and why:
- `ultralytics`, `opencv-python`, `numpy` — the CV/model core.
- `mlflow`, `pandas`, `pyarrow` — local model packaging + tabular IO.
- `pyspark` + `delta-spark` + `openjdk` — exercise the `pandas_udf` / Delta `MERGE`
  path locally (Spark 3.5 / Delta 3.2 mirror Databricks Runtime ML 15.x).
- `databricks-sdk` — importable locally; full features need a workspace.

> The Databricks-only packages (`databricks-feature-engineering`, serving, monitoring)
> live in `requirements.txt` and are installed on the cluster, not here.

---

## Step 1 — extract frames locally (the ingest core)

`local_dev/extract_frames_local.py` is the portable half of `01_ingest`: same
`frame_id` convention, same manifest columns, but local disk + Parquet instead of
a Delta table.

```bash
conda activate yolo-databricks
cd local_dev
python extract_frames_local.py \
  --video ../../posts/02-volei-brasil-italia/media/brasil-italia-nDeXUywqnIs-clip.mp4 \
  --interval 30 --max-frames 12
```

Result: **12 frames @ 1920×1080**, sampled every 30 frames (≈ every 0.5 s on the
60 fps clip), written to `local_dev/_work/frames/` with a `frames.parquet` manifest
carrying the exact canonical schema (`video_id, frame_id, frame_path, timestamp_ms,
width, height`). `frame_id` example: `brasil-italia-nDeXUywqnIs-clip__000270`.

## Step 2 — validate the YOLOPyfunc contract locally

`local_dev/test_pyfunc_contract.py` imports the **real** `YOLOPyfunc` class (no
copy) and runs it over the manifest, asserting the pipeline-wide contract:

```bash
python test_pyfunc_contract.py   # reuses ../../volei-tactical/yolov8n.pt
```

Result: contract holds — `predict` returns exactly `[frame_id, boxes, confs,
classes]`, row order preserved, each box is `{x1,y1,x2,y2,conf,cls}` with the right
types. **266 detections across 12 frames**; an annotated frame is saved to
`_work/annotated/`.

> **Insight for the write-up:** the off-the-shelf COCO `yolov8n.pt` detects *every*
> person — players, bench, referee, and the crowd. That's exactly why stage 03
> trains a custom model on a court-filtered dataset: the value isn't "detect people",
> it's "detect the players that matter". The local run makes that gap visible.

## Step 3 — package the model with MLflow (log → load → predict), locally

`local_dev/test_mlflow_packaging.py` runs the exact two MLflow calls the pipeline
depends on — `log_model(python_model=YOLOPyfunc(), artifacts={"weights": ...},
code_paths=[...])` (stage 03) and `load_model(...).predict(df)` (stage 05) — but
against a **local sqlite tracking backend**, not the Unity Catalog registry.

```bash
python test_mlflow_packaging.py
```

Result: model logged, reloaded from `runs:/<id>/model`, and the reloaded pyfunc
produced the **same 266 detections across 12 frames**. So the packaging path
(cloudpickle + `weights` artifact + `code_paths`) is sound — the only thing left
for Databricks is the registry/UC wiring, not the model itself.

Notes captured from the run (worth knowing before the cluster):
- Recent MLflow puts the bare-file tracking store in *maintenance mode*; use a
  `sqlite:///` (or real) backend locally. On Databricks this is moot — the workspace
  provides the tracking server.
- A dependency mismatch is reported locally (`opencv-python-headless` "uninstalled")
  because the dev env uses the GUI build of OpenCV; the cluster uses `*-headless`.
  Harmless locally.
- MLflow could not infer a model **signature** (our `predict` is typed only as
  `DataFrame -> DataFrame`). **Hardening TODO for `train.py`:** log an explicit
  `signature` + `input_example` so the registered model self-documents its IO.

## Step 4 — distributed inference locally: Spark `pandas_udf` + Delta MERGE

`local_dev/test_spark_udf_delta.py` reproduces `05_inference/batch_inference.py`
against a **local Spark + Delta** (the real file can't be imported locally — it reads
`dbutils` widgets and three-part UC names at import). Same logic: a `pandas_udf`
typed `array<struct<...>>` wrapping the pyfunc, a LEFT ANTI JOIN for incremental
work, and a Delta `MERGE` by `frame_id`. It runs twice to prove idempotency.

```bash
python test_spark_udf_delta.py
```

Result:
- Pass 1 (cold): created the Delta detections table, scored **12 frames**.
- Pass 2: **0 unprocessed frames** — idempotency proven (anti-join + MERGE).
- Delta table: 12 rows, the `array<struct>` materialized correctly; `n_detections`
  and `avg_conf` per frame computed with pure Spark (`size`, `aggregate`).

This is the whole pipeline running on a laptop: **ingest → detect → distributed
inference → Delta table**, before any cluster exists.

### Bug caught locally (and fixed in the real code) 🐛

The first run **crashed** with:

```
PythonException: ModuleNotFoundError: No module named 'yolo_pyfunc'
  at broadcast_model.value  (pyspark/broadcast.py, load)
```

Root cause: `batch_inference.py` loaded the pyfunc on the driver and **broadcast**
it to the executors. Broadcasting pickles the custom `YOLOPyfunc` class, which a
worker process can't unpickle unless that class' code is already importable there —
and it isn't.

Fix (applied to `05_inference/batch_inference.py`): **don't broadcast a loaded
model**. Instead, load it lazily *on each worker* via `mlflow.pyfunc.load_model`,
cached in a module-level dict. `load_model` ships the model's logged `code_paths`,
so the class is always importable, and the cache keeps it to one load per worker:

```python
_WORKER_MODEL: dict = {}

@pandas_udf(BOXES_RETURN_TYPE)
def detect_udf(paths: pd.Series) -> pd.Series:
    if "model" not in _WORKER_MODEL:
        _WORKER_MODEL["model"] = mlflow.pyfunc.load_model(model_uri)  # ships code_paths
    model = _WORKER_MODEL["model"]
    ...
```

> This is the payoff of testing locally first: a distribution bug that would have
> only surfaced on a multi-worker cluster was found and fixed on the laptop.

### Hardening applied to `train.py`

MLflow couldn't infer a signature from type hints (Step 3 note), so `train.py` now
logs a **best-effort model signature + `input_example`** (guarded by try/except so
it never breaks a training run). Verified locally — the inferred output schema is
exactly `frame_id: string, boxes: Array(struct), confs: Array(double), classes:
Array(long)`.

### Version note (local vs cluster)

The local env resolved **MLflow 3.x** (latest for `mlflow>=2.14`). Databricks
Runtime ML 15.x ships **MLflow 2.x**. The APIs used here exist in both; in
particular `log_model(artifact_path=...)` is kept (correct on DBR 2.x) even though
MLflow 3.x prefers `name=`. Nothing to change for the cluster.

---

## ✅ Local validation complete

The portable core is fully proven on the laptop. What remains is **Databricks
infrastructure only** — not pipeline logic:

| Remaining (Databricks-only) | Why it needs a workspace |
|---|---|
| Unity Catalog `catalog.schema` + Volumes | UC + `/Volumes` are workspace constructs |
| Feature Engineering client (`02`) | needs the online/offline store |
| UC model registry + `@champion` (`04`) | `models:/...@alias` is UC-backed |
| Model Serving endpoint (`05`, optional) | billed workspace resource |
| Lakehouse Monitoring (`06`) | workspace quality-monitor service |

Next: stand up the Databricks side (catalog/schema/volume, dataset upload, Jobs).

---

## Databricks phase — inference-only first (no labels yet)

We don't have a labeled dataset yet, so we **defer custom training** and run the
pipeline end-to-end on Databricks with the **pretrained** `yolov8n.pt`. The new
`03_train/register_pretrained.py` wraps those COCO weights in the same `YOLOPyfunc`,
registers the model in Unity Catalog, and sets `@champion` — so `05_inference` works
immediately. `train.py` + `promote_model.py` come back once annotations exist; the
registered-model + alias contract is identical, so nothing downstream changes.

➡️ Full guided runbook: [`DATABRICKS.md`](DATABRICKS.md) (workspace UI / Repos, step
by step: UC objects → upload video → ingest → features → register pretrained →
batch inference → monitoring).

<!-- STEP LOG: append each local/Databricks step below as it is run and verified. -->

