# Databricks runbook — inference-only path (pretrained weights, no labels yet)

Guided, workspace-driven steps to run the pipeline end-to-end on Databricks using
the **pretrained** `yolov8n.pt` (COCO `person` detection). No labeled dataset is
required. When annotations exist later, swap the "register pretrained" step for
`03_train/train.py` + `04_registry/promote_model.py` — nothing else changes.

Pipeline for this path:

```
01_ingest → 02_feature_store → 03_train/register_pretrained.py → 05_inference → 06_monitoring
                                  (replaces train.py + promote_model.py)
```

## Prerequisites

- A Databricks workspace with **Unity Catalog** enabled.
- A cluster on **Databricks Runtime ML 15.x**. CPU is fine for inference-only with
  `yolov8n`; a GPU cluster is only needed for custom training later.
- Permission to create a catalog/schema/volume (or use existing ones you can write to).

Pick your coordinates (used as widget values everywhere below):

| Setting | Example |
|---|---|
| catalog | `main` |
| schema | `volei_tactical` |
| model_name | `yolo_volei` |

---

## D1 — Create the Unity Catalog objects

In a SQL cell or the SQL editor (replace `<catalog>`/`<schema>`):

```sql
CREATE CATALOG IF NOT EXISTS <catalog>;
CREATE SCHEMA  IF NOT EXISTS <catalog>.<schema>;

-- Volume for raw match videos (ingest input)
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.raw_videos;
-- Volume for the dataset + extracted frames (ingest output, training input)
CREATE VOLUME IF NOT EXISTS <catalog>.<schema>.dataset;
```

## D2 — Get the code into the workspace

- Push this `yolo_databricks_mlops/` folder to GitHub, then in Databricks:
  **Workspace → Repos → Add Repo** and clone it. (Or **Workspace → Import** the
  `.py` files — each has the `# Databricks notebook source` header and runs as a notebook.)

## D3 — Upload the video

- **Catalog → `<catalog>` → `<schema>` → `raw_videos` → Upload to volume**, and
  upload `brasil-italia-nDeXUywqnIs-clip.mp4` (from `posts/02-volei-brasil-italia/media/`).
- Target path: `/Volumes/<catalog>/<schema>/raw_videos/`.

## D4 — Cluster libraries

On the cluster (or as the first notebook cell), install the extra deps:

```python
%pip install -r ../requirements.txt
dbutils.library.restartPython()
```

> `mlflow` and `pyspark` already ship with DBR ML; `requirements.txt` adds
> `ultralytics`, `databricks-feature-engineering`, `databricks-sdk`,
> `opencv-python-headless`.

## D5 — Run `01_ingest/ingest_frames.py`

Widgets:

| widget | value |
|---|---|
| `catalog` | `<catalog>` |
| `schema` | `<schema>` |
| `frame_interval` | `30` |

Produces `<catalog>.<schema>.frames` (metadata) and frame JPEGs under
`/Volumes/<catalog>/<schema>/dataset/frames/`.

## D6 — Run `02_feature_store/register_features.py`

Widgets: `catalog`, `schema`. Creates the feature table
`<catalog>.<schema>.frame_features` (PK `frame_id`).

## D7 — Register the pretrained model: `03_train/register_pretrained.py`

This replaces training+promotion for the no-labels path. Widgets:

| widget | value |
|---|---|
| `catalog` | `<catalog>` |
| `schema` | `<schema>` |
| `model_name` | `yolo_volei` |
| `base_model` | `yolov8n.pt` |
| `alias` | `champion` |

Registers `<catalog>.<schema>.yolo_volei` and sets `@champion`.

> **Air-gapped workspace?** If the cluster has no internet egress, Ultralytics
> can't download `yolov8n.pt`. Upload `volei-tactical/yolov8n.pt` to a volume and
> set `base_model` to its `/Volumes/...` path.

## D8 — Run `05_inference/batch_inference.py`

Widgets:

| widget | value |
|---|---|
| `catalog` | `<catalog>` |
| `schema` | `<schema>` |
| `model_name` | `yolo_volei` |
| `alias` | `champion` |

Loads `models:/<catalog>.<schema>.yolo_volei@champion`, scores unprocessed frames
(LEFT ANTI JOIN), and MERGEs into `<catalog>.<schema>.detections`. Re-running is
idempotent (we proved this locally in WALKTHROUGH Step 4).

## D9 — Run `06_monitoring/setup_monitoring.py`

Widgets: `catalog`, `schema`. Flattens detections into
`<catalog>.<schema>.detections_metrics` (`n_detections`, `avg_conf`), creates a
baseline, and sets up a Lakehouse Monitor with `1 day` granularity. Schedule the
refresh via a Job (see the in-file comment).

## D10 — (Optional, billed) Serving endpoint

`05_inference/endpoint_serving.py` is gated by `RUN = False`. Only flip it if you
want a live REST endpoint — it bills per hour. Put the PAT in a secret scope
(`volei`), never in code.

---

## Creating Jobs (instead of running interactively)

For each notebook, **Workflows → Create Job → Notebook task**, point it at the file,
and add the widgets above as **Job parameters**. A sensible single multi-task Job:

```
ingest → register_features → register_pretrained → batch_inference → setup_monitoring
```

(each task depends on the previous). Training/promotion tasks are added later, once
a labeled dataset exists.

## What we skipped and why

- `03_train/train.py` and `04_registry/promote_model.py` — these need a **labeled**
  YOLO dataset (`data.yaml` + `images/` + `labels/`). We start inference-only with
  pretrained weights; revisit once annotations exist.
