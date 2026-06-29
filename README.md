# yolo_databricks_mlops

Production MLOps pipeline on **Databricks** for **volleyball position analysis** with
**YOLOv8**. This repository takes the local **`volei-tactical`** proof of concept and
productionizes it on Databricks using **Unity Catalog**, **MLflow**, and **Delta Lake**.

---

## Overview

`volei-tactical` is a local computer-vision PoC: it runs YOLOv8 to detect volleyball
players on an **18 x 9 m court**, applies a pixel-to-meters homography, builds a Delaunay
triangulation over player positions, and separates the two teams by jersey color to
produce tactical position analysis.

This project lifts that PoC into a reproducible, governed production pipeline:

- **Unity Catalog** for governed data and model assets — every table, volume, and model
  uses a three-part `catalog.schema.object` namespace.
- **Delta Lake** for idempotent, MERGE-based storage of frames, features, and detections.
- **MLflow (UC registry)** for experiment tracking and model versioning, promoting models
  with the `@champion` / `@challenger` aliases (no legacy Staging/Production stages).
- **Spark-first** batch processing; pandas is used only inside UDFs.

The pipeline ingests raw match videos, extracts frames, engineers per-frame features,
trains a YOLOv8 model packaged as an MLflow `pyfunc`, registers and promotes it in Unity
Catalog, runs batch inference to produce player detections, and monitors data and model
quality over time.

---

## Prerequisites

- **Databricks Runtime ML 15.x** (14.x+ supported; 15.x recommended).
- **Unity Catalog enabled** on the workspace, with a metastore attached and permission to
  create a catalog/schema (or an existing one you can write to).
- A **GPU cluster** for the training step (`03_train`). The remaining steps run on
  CPU clusters.
- Permission to create **Unity Catalog Volumes** for raw videos and the training dataset.
- Cluster libraries from [`requirements.txt`](./requirements.txt). At the top of each
  notebook/job, install them with:

  ```python
  %pip install -r requirements.txt
  dbutils.library.restartPython()
  ```

  Key dependencies: `ultralytics` (YOLOv8), `databricks-feature-engineering`,
  `databricks-sdk`, `opencv-python-headless`. `mlflow` and `pyspark` ship with the runtime.

---

## Unity Catalog and Volume setup

All objects live under a three-part namespace. Pick your own `catalog` and `schema`
values and use them consistently across every job widget (see below). The examples use
the placeholders `<catalog>` and `<schema>`.

### 1. Create the catalog, schema, and volumes

Run once in a Databricks SQL editor or notebook:

```sql
-- Catalog and schema
CREATE CATALOG IF NOT EXISTS <catalog>;
CREATE SCHEMA  IF NOT EXISTS <catalog>.<schema>;

-- Volume for raw match videos (ingestion input)
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.raw_videos;

-- Volume for the YOLO training dataset and extracted frames
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.dataset;
```

This creates the canonical volume paths used by the pipeline:

- Raw videos: `/Volumes/<catalog>/<schema>/raw_videos/`
- Dataset:    `/Volumes/<catalog>/<schema>/dataset/`

> Always use `/Volumes/...` paths. `dbfs:/` is **not** used for new data.

### 2. Upload the raw videos

Copy your match videos into the raw videos volume:

```
/Volumes/<catalog>/<schema>/raw_videos/
├── match_001.mp4
├── match_002.mp4
└── ...
```

### 3. Upload the YOLO training dataset

The training step expects a standard Ultralytics YOLO dataset layout rooted at the
dataset volume, with a `data.yaml` manifest plus `images/` and `labels/` directories
split into `train/` and `val/`:

```
/Volumes/<catalog>/<schema>/dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── val/
│       ├── frame_1001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── frame_0001.txt
    │   └── ...
    └── val/
        ├── frame_1001.txt
        └── ...
```

The canonical dataset manifest path is
`/Volumes/<catalog>/<schema>/dataset/data.yaml`. A typical `data.yaml` for volleyball
position analysis looks like:

```yaml
# data.yaml — Ultralytics dataset manifest (volleyball position analysis)
path: /Volumes/<catalog>/<schema>/dataset
train: images/train
val: images/val
names:
  0: player
  1: ball
  2: referee
```

---

## Data and model assets (canonical identifiers)

These names are part of the project contract — other scripts depend on them exactly:

| Asset            | Identifier                                              | Notes                                                                 |
| ---------------- | ------------------------------------------------------- | --------------------------------------------------------------------- |
| Frames table     | `<catalog>.<schema>.frames`                             | Merge key / logical PK: `frame_id`                                    |
| Feature table    | `<catalog>.<schema>.frame_features`                     | Primary key: `frame_id`                                              |
| Detections table | `<catalog>.<schema>.detections`                         | Merged by `frame_id`                                                 |
| Registered model | `<catalog>.<schema>.<model_name>`                       | UC model; promoted with `@champion` / `@challenger` aliases          |
| Raw videos       | `/Volumes/<catalog>/<schema>/raw_videos/`               | Ingestion input                                                       |
| Dataset          | `/Volumes/<catalog>/<schema>/dataset/data.yaml`         | YOLO training manifest                                                |
| MLflow experiment| `/Users/<user>/yolo-cv`                                 | Set the registry URI to `databricks-uc`                              |

**`frame_id` convention:** `f"{video_id}__{frame_number:06d}"` — two underscores between
the fields, frame number zero-padded to six digits (e.g. `match_001__000042`).

**`frames` table schema:** `video_id STRING`, `frame_id STRING`, `frame_path STRING`,
`timestamp_ms LONG`, `width INT`, `height INT`, `extracted_at TIMESTAMP`.

---

## Recommended execution order (01 → 06)

Run the pipeline steps in numeric order. Each directory holds one standalone Databricks
script/notebook.

| Order | File                                  | What it does                                                                                                                |
| ----- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 01    | `01_ingest/ingest_frames.py`          | Reads videos from the `raw_videos` volume, extracts frames to the `dataset` volume, and MERGEs frame metadata into `frames`.|
| 02    | `02_feature_store/register_features.py` | Computes per-frame features and writes the `frame_features` feature table (keyed by `frame_id`) via Feature Engineering.   |
| 03    | `03_train/train.py`                   | Trains YOLOv8 on the dataset volume and logs the run + `YOLOPyfunc` model to the MLflow experiment.                          |
| 03    | `03_train/yolo_pyfunc.py`             | Defines `YOLOPyfunc(mlflow.pyfunc.PythonModel)` — the inference wrapper used at logging and serving time.                    |
| 04    | `04_registry/promote_model.py`        | Registers the trained model in Unity Catalog and assigns the `@challenger` / `@champion` aliases after validation.           |
| 05    | `05_inference/batch_inference.py`     | Loads the model by alias, runs a Spark `pandas_udf` over `frames`, and MERGEs results into `detections`.                     |
| 06    | `06_monitoring/setup_monitoring.py`   | Tracks detection volume, score distributions, and drift; can flag the active alias for review.                              |

> `03_train/yolo_pyfunc.py` is a library module imported by training and inference — it is
> not run as a standalone job, but it must be packaged with the logged model.

### pyfunc ↔ inference contract

`YOLOPyfunc.predict` accepts a pandas `DataFrame` with a `"path"` column (string) and
returns a pandas `DataFrame` with the exact columns `"frame_id"`, `"boxes"`, `"confs"`,
`"classes"`. `"boxes"` is a per-frame list of structs
`{x1: float, y1: float, x2: float, y2: float, conf: float, cls: int}`.

`batch_inference.py` declares a `@pandas_udf` returning
`array<struct<x1:float,y1:float,x2:float,y2:float,conf:float,cls:int>>` and loads the
model with:

```python
mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(
    f"models:/{catalog}.{schema}.{model_name}@{alias}"
)
```

The `detections` table additionally carries `model_version STRING`,
`inferred_at TIMESTAMP`, and `alias_used STRING`.

---

## Creating the Databricks Jobs

Create one Job task per step (Workflows → Jobs → Add task → Type: Notebook/Python),
attach the appropriate cluster, and define the task parameters as **widgets**. Chain the
tasks `01 → 02 → 03 → 04 → 05 → 06` with task dependencies. Use the **GPU cluster** only
for the `03_train` task.

The widgets below are read with `dbutils.widgets.get(...)` inside each script. Set
`catalog` and `schema` to the values you created above on **every** task.

### 01 — `01_ingest/ingest_frames.py`

| Widget           | Example   | Purpose                                            |
| ---------------- | --------- | -------------------------------------------------- |
| `catalog`        | `main`    | Unity Catalog catalog.                             |
| `schema`         | `volei`   | Unity Catalog schema.                              |
| `frame_interval` | `15`      | Sampling stride: save one frame every N frames.    |

### 02 — `02_feature_store/register_features.py`

| Widget    | Example | Purpose                                  |
| --------- | ------- | ---------------------------------------- |
| `catalog` | `main`  | Unity Catalog catalog.                   |
| `schema`  | `volei` | Unity Catalog schema.                    |

### 03 — `03_train/train.py`  *(GPU cluster)*

| Widget         | Example       | Purpose                                                        |
| -------------- | ------------- | -------------------------------------------------------------- |
| `catalog`      | `main`        | Unity Catalog catalog.                                         |
| `schema`       | `volei`       | Unity Catalog schema.                                          |
| `model_name`   | `yolo_volei`  | UC registered model name (`catalog.schema.model_name`).        |
| `epochs`       | `100`         | Training epochs.                                               |
| `imgsz`        | `640`         | Training image size.                                           |
| `batch_size`   | `16`          | Training batch size.                                           |
| `base_model`   | `yolov8n.pt`  | YOLOv8 base weights to fine-tune (set as the `base_model` tag).|

> The run is tagged with `{"project": "volei-tactical", "base_model": base_model}`. The
> MLflow experiment path is **not** a widget — it is derived as `/Users/<user>/yolo-cv`
> from the current user.

### 04 — `04_registry/promote_model.py`

| Widget            | Example                 | Purpose                                                          |
| ----------------- | ----------------------- | ---------------------------------------------------------------- |
| `catalog`         | `main`                  | Unity Catalog catalog.                                           |
| `schema`          | `volei`                 | Unity Catalog schema.                                            |
| `model_name`      | `yolo_volei`            | UC registered model name (`catalog.schema.model_name`).          |
| `experiment_path` | `/Users/<user>/yolo-cv` | Optional MLflow experiment to source the run from.               |
| `min_gain`        | `1.02`                  | Optional override for the relative-gain gate (e.g. `1.02` = +2%).|

### 05 — `05_inference/batch_inference.py`

| Widget         | Example      | Purpose                                                           |
| -------------- | ------------ | ----------------------------------------------------------------- |
| `catalog`      | `main`       | Unity Catalog catalog.                                            |
| `schema`       | `volei`      | Unity Catalog schema.                                             |
| `model_name`   | `yolo_volei` | UC registered model name.                                         |
| `alias`        | `champion`   | Model alias to load (`champion` or `challenger`).                 |

> Detection confidence / IoU / image size are **baked into the model** at
> registration (`PREDICT_CONF_THRESHOLD` etc. in `03_train/yolo_pyfunc.py`), not
> passed as inference widgets.

### 06 — `06_monitoring/setup_monitoring.py`

| Widget    | Example | Purpose                  |
| --------- | ------- | ------------------------ |
| `catalog` | `main`  | Unity Catalog catalog.   |
| `schema`  | `volei` | Unity Catalog schema.    |

---

## Secrets and variables

This pipeline reads data from Unity Catalog Volumes and authenticates to the workspace
through the cluster's own identity, so **no token is required for the core 01 → 06 flow**.

A token is only needed for the **optional** real-time serving endpoint (a streaming/online
variant of batch inference). When used, fetch it via `dbutils.secrets` — never hardcode it:

```python
# Create the scope and secret once (Databricks CLI), then read at runtime:
#   databricks secrets create-scope volei
#   databricks secrets put-secret volei serving_token
token = dbutils.secrets.get(scope="volei", key="serving_token")
```

| Secret (scope/key)        | Required?            | Used by                                  |
| ------------------------- | -------------------- | ---------------------------------------- |
| `volei` / `serving_token` | Optional             | Optional model serving endpoint only.    |

Hardcoded tokens or credentials are forbidden anywhere in the codebase.

---

## Configuration to adjust (`# CONFIGURE`)

Every script keeps its tunable, environment-specific values either in **`dbutils.widgets`**
(see the job tables above) or in **top-of-file variables tagged with the exact comment
`# CONFIGURE`**. Review and set these before running:

- **`01_ingest/ingest_frames.py`** — `# CONFIGURE`: default `catalog` / `schema`, the
  default frame-sampling stride, and the JPEG quality used when encoding frames.
- **`02_feature_store/register_features.py`** — `# CONFIGURE`: default `catalog` / `schema`
  and the `frame_features` feature-table settings.
- **`03_train/train.py`** — `# CONFIGURE`: UC registry URI, the MLflow experiment root /
  name that derive `/Users/<user>/yolo-cv`, project tag, artifact path, scratch output
  dir, and the pip requirements baked into the logged model.
- **`03_train/yolo_pyfunc.py`** — `# CONFIGURE`: default `base_model` weights bundled with
  the `YOLOPyfunc` artifact.
- **`04_registry/promote_model.py`** — `# CONFIGURE`: the minimum relative gain gating
  `@challenger` → `@champion` and the ranking metric used to compare runs.
- **`05_inference/batch_inference.py`** — `# CONFIGURE`: default `catalog` / `schema`,
  `model_name`, and default `alias`.
- **`06_monitoring/setup_monitoring.py`** — `# CONFIGURE`: default `catalog` / `schema`,
  the baseline monitoring window in days, and the monitor granularity.

Search the repository for the marker to find every spot:

```bash
grep -rn "# CONFIGURE" .
```
