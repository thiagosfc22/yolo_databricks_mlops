# Databricks notebook source
"""Optional real-time Model Serving endpoint for the YOLOv8 volleyball model.

Pipeline position
-----------------
Stage 05 (inference), OPTIONAL branch. The canonical, cost-free path for this
project is the batch scorer ``05_inference/batch_inference.py``, which loads the
UC model with ``mlflow.pyfunc.load_model`` and writes the ``detections`` Delta
table. THIS file is an *alternative* low-latency path that exposes the same
registered model behind a Databricks Model Serving REST endpoint.

COST WARNING
------------
A Model Serving endpoint bills PER HOUR for as long as it exists, even when it is
not actively serving requests (scale-to-zero reduces, but does NOT eliminate,
idle cost, and there is cold-start latency on the first request after scaling in).
This endpoint is NOT required by the batch pipeline and should only be created
when real-time / on-demand single-frame scoring is genuinely needed. To avoid
accidental spend, all create/update calls below are guarded by ``RUN = False``.
Flip ``RUN`` to ``True`` only when you intentionally want to provision (and pay
for) the endpoint.

Contract
--------
The served model is the same ``YOLOPyfunc`` registered in Unity Catalog by the
training/registry stages. Its ``predict`` consumes a pandas DataFrame with a
single string column ``"path"`` and returns the columns
``"frame_id", "boxes", "confs", "classes"`` (``boxes`` being a per-frame list of
structs ``{x1, y1, x2, y2, conf, cls}``). The REST payload therefore mirrors that
input: a ``dataframe_records`` list of ``{"path": "/Volumes/.../frame.jpg"}``.

Requirements
------------
- Databricks Runtime ML 15.x.
- Unity Catalog model registry (``databricks-uc``) with a ``@champion`` alias.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

# COMMAND ----------

# -------------------------------------------------------------------------
# Configuration. Everything is parameterized via widgets / top-of-file vars.
# Nothing is hardcoded: paths, catalog/schema, model name and the cost guard
# are all exposed below.
# -------------------------------------------------------------------------

dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")  # CONFIGURE
dbutils.widgets.text("schema", "volei_tactical", "Unity Catalog schema")  # CONFIGURE
dbutils.widgets.text("model_name", "yolo_volei", "UC registered model name")  # CONFIGURE
dbutils.widgets.text("endpoint_name", "yolo-volei-serving", "Serving endpoint name")  # CONFIGURE
dbutils.widgets.text("model_alias", "champion", "UC alias to serve (champion/challenger)")  # CONFIGURE

catalog: str = dbutils.widgets.get("catalog")
schema: str = dbutils.widgets.get("schema")
model_name: str = dbutils.widgets.get("model_name")
endpoint_name: str = dbutils.widgets.get("endpoint_name")
model_alias: str = dbutils.widgets.get("model_alias")

# Three-part Unity Catalog namespace for the registered model.
uc_model: str = f"{catalog}.{schema}.{model_name}"

# Serving sizing. "Small" is the cheapest tier; scale_to_zero spins the endpoint
# down to zero compute when idle to minimize (but not eliminate) hourly cost.
WORKLOAD_SIZE: str = "Small"  # CONFIGURE
SCALE_TO_ZERO_ENABLED: bool = True  # CONFIGURE

# MASTER COST GUARD. While False, this script provisions nothing and bills
# nothing. Set to True ONLY when you intentionally want to create/update the
# (hourly-billed) endpoint.
RUN: bool = False  # CONFIGURE

# Secret scope/key used to read a PAT for the example REST invocation. NEVER
# hardcode tokens; they are fetched at runtime via dbutils.secrets.
SECRET_SCOPE: str = "volei"  # CONFIGURE
SECRET_TOKEN_KEY: str = "serving_token"  # CONFIGURE

# Example frame path used only by the illustrative invocation below.
EXAMPLE_FRAME_PATH: str = f"/Volumes/{catalog}/{schema}/raw_videos/example_frame.jpg"  # CONFIGURE

# COMMAND ----------


def build_endpoint_config(
    name: str,
    entity: str,
    alias: str,
    workload_size: str = WORKLOAD_SIZE,
    scale_to_zero_enabled: bool = SCALE_TO_ZERO_ENABLED,
) -> EndpointCoreConfigInput:
    """Build the serving endpoint core configuration for the UC model.

    Args:
        name: Name of the serving endpoint.
        entity: Fully qualified UC model name (``catalog.schema.model_name``).
        alias: UC alias to serve (e.g. ``"champion"`` or ``"challenger"``).
            The endpoint always tracks the model version currently pointed to
            by this alias, so promoting a new champion auto-updates serving.
        workload_size: Serving compute tier; ``"Small"`` is the cheapest.
        scale_to_zero_enabled: When ``True``, the endpoint scales down to zero
            compute (and cost) while idle, at the price of cold-start latency.

    Returns:
        An ``EndpointCoreConfigInput`` describing a single served entity that
        tracks the requested UC alias.
    """
    served_entity = ServedEntityInput(
        name=model_name,
        entity_name=entity,
        # Serve "by alias" so champion promotions propagate without redeploy.
        entity_version=None,
        entity_alias=alias,
        workload_size=workload_size,
        scale_to_zero_enabled=scale_to_zero_enabled,
    )
    return EndpointCoreConfigInput(
        name=name,
        served_entities=[served_entity],
    )


def create_or_update_endpoint(
    w: WorkspaceClient,
    name: str,
    entity: str,
    alias: str,
) -> None:
    """Create the serving endpoint, or update it if it already exists.

    This call provisions hourly-billed compute. It is intentionally only ever
    reached when the ``RUN`` cost guard is ``True``.

    Args:
        w: Authenticated Databricks ``WorkspaceClient``.
        name: Name of the serving endpoint.
        entity: Fully qualified UC model name (``catalog.schema.model_name``).
        alias: UC alias to serve (e.g. ``"champion"``).
    """
    config = build_endpoint_config(name=name, entity=entity, alias=alias)

    existing_names = [e.name for e in w.serving_endpoints.list()]
    if name in existing_names:
        # Endpoint already exists -> push the new config and wait for readiness.
        print(f"[serving] Updating existing endpoint '{name}' to serve {entity}@{alias} ...")
        w.serving_endpoints.update_config_and_wait(
            name=name,
            served_entities=config.served_entities,
        )
    else:
        # First-time provisioning -> create and wait until it is READY.
        print(f"[serving] Creating endpoint '{name}' to serve {entity}@{alias} ...")
        w.serving_endpoints.create_and_wait(
            name=name,
            config=config,
        )
    print(f"[serving] Endpoint '{name}' is ready.")


# COMMAND ----------

# -------------------------------------------------------------------------
# Guarded provisioning. Nothing below runs (and nothing is billed) unless the
# RUN cost guard is explicitly flipped to True.
# -------------------------------------------------------------------------

if RUN:
    w = WorkspaceClient()
    create_or_update_endpoint(
        w=w,
        name=endpoint_name,
        entity=uc_model,
        alias=model_alias,
    )
else:
    print(
        "[serving] RUN is False -> no endpoint created/updated and NO hourly cost "
        "incurred. Set RUN = True (CONFIGURE) only when you intentionally want to "
        "provision the (per-hour billed) Model Serving endpoint."
    )

# COMMAND ----------


def invoke_endpoint(
    host: str,
    name: str,
    token: str,
    records: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Call the serving endpoint over REST with the contract payload.

    The payload mirrors the ``YOLOPyfunc`` input contract: a list of records,
    each holding a single ``"path"`` string pointing at a frame on a Unity
    Catalog Volume. The response carries the predictions produced by the model
    (frame_id / boxes / confs / classes).

    Args:
        host: Workspace base URL, e.g. ``https://<workspace>.cloud.databricks.com``.
        name: Name of the serving endpoint to invoke.
        token: Bearer token (PAT) read from a secret scope; NEVER hardcoded.
        records: List of ``{"path": "/Volumes/.../frame.jpg"}`` input records.

    Returns:
        The parsed JSON response from the serving endpoint.
    """
    url = f"{host}/serving-endpoints/{name}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"dataframe_records": records}
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
    response.raise_for_status()
    return response.json()


# COMMAND ----------

# -------------------------------------------------------------------------
# Example REST invocation. Also guarded by RUN, since calling a live endpoint
# requires it to be provisioned (and therefore billing). Credentials are read
# from dbutils.secrets at runtime -- no tokens are ever written in this file.
# -------------------------------------------------------------------------

if RUN:
    # Workspace host resolved from the notebook context (no hardcoded URL).
    host = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .apiUrl()
        .get()
    )
    # PAT pulled from a secret scope -- NEVER hardcode tokens.
    token = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_TOKEN_KEY)

    example_records: List[Dict[str, str]] = [{"path": EXAMPLE_FRAME_PATH}]
    result = invoke_endpoint(
        host=host,
        name=endpoint_name,
        token=token,
        records=example_records,
    )
    print("[serving] Example invocation response:")
    print(json.dumps(result, indent=2))
else:
    # Documentation-only example of the exact REST shape, so readers can see the
    # contract without provisioning anything:
    #
    #   POST {host}/serving-endpoints/{endpoint_name}/invocations
    #   Authorization: Bearer <token-from-dbutils.secrets>
    #   Content-Type: application/json
    #
    #   {"dataframe_records": [{"path": "/Volumes/<catalog>/<schema>/raw_videos/example_frame.jpg"}]}
    #
    print(
        "[serving] RUN is False -> skipping the live REST invocation (it would "
        "require a running, billed endpoint). See the comment above for the exact "
        "payload shape."
    )
