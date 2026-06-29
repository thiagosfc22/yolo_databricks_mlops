# Databricks notebook source
"""Model promotion gate for the volei-tactical YOLOv8 pipeline (stage 04_registry).

Pipeline position
-----------------
This is the registry / promotion step. It runs AFTER ``03_train`` has logged
training runs to the MLflow experiment ``/Users/{user}/yolo-cv`` and registered
candidate versions of the Unity Catalog model ``{catalog}.{schema}.{model_name}``.
It runs BEFORE ``05_inference``, which loads the model exclusively through the UC
alias ``@champion`` (and may shadow-test ``@challenger``).

What it does
------------
1. Lists the experiment runs ordered by ``metrics.mAP50`` (descending).
2. Identifies the best run and its corresponding registered model version.
3. Reads the mAP50 of the incumbent ``@champion`` version, if one exists
   (this is a no-op safe path on the very first deploy).
4. Promotes the best candidate to ``@champion`` ONLY when its mAP50 beats the
   incumbent by at least ``MIN_GAIN`` (a relative-gain gate); otherwise the
   current champion is kept.
5. Assigns the second-best version to the ``@challenger`` alias for ongoing
   comparison / shadow inference.
6. Logs the promotion decision as a tag on the winning run for auditability.

Conventions honoured
---------------------
- Unity Catalog three-part namespace for every object (``catalog.schema.object``).
- MLflow Registry URI set to ``databricks-uc`` (no legacy Workspace registry).
- UC aliases ``@champion`` / ``@challenger`` instead of legacy
  Staging/Production stages.
- Zero hardcoded paths/credentials: configuration arrives via ``dbutils.widgets``
  or top-of-file variables tagged with the exact comment ``# CONFIGURE``.
"""

from __future__ import annotations

from typing import Optional

import mlflow
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

# COMMAND ----------

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Relative-gain gate: a challenger only becomes champion if its mAP50 exceeds the
# incumbent champion's mAP50 by this multiplicative factor (1.02 == +2% minimum).
MIN_GAIN: float = 1.02  # CONFIGURE (minimum relative gain to promote, e.g. 1.02 == +2%)

# Metric used both to rank runs and to gate the promotion decision.
RANKING_METRIC: str = "mAP50"  # CONFIGURE (run metric name logged during training)

# COMMAND ----------

# -----------------------------------------------------------------------------
# Widgets (job/notebook parameters) — keep ZERO hardcoded values here.
# -----------------------------------------------------------------------------
dbutils.widgets.text("catalog", "", "Unity Catalog catalog")
dbutils.widgets.text("schema", "", "Unity Catalog schema")
dbutils.widgets.text("model_name", "", "Registered model name (UC object)")
# Optional widgets: sensible defaults are derived below when left blank.
dbutils.widgets.text("experiment_path", "", "MLflow experiment path (optional)")
dbutils.widgets.text("min_gain", "", "Override for the relative-gain gate (optional)")

catalog: str = dbutils.widgets.get("catalog").strip()
schema: str = dbutils.widgets.get("schema").strip()
model_name: str = dbutils.widgets.get("model_name").strip()
experiment_path_widget: str = dbutils.widgets.get("experiment_path").strip()
min_gain_widget: str = dbutils.widgets.get("min_gain").strip()

if not catalog or not schema or not model_name:
    raise ValueError(
        "Widgets 'catalog', 'schema' and 'model_name' are all required and must be non-empty."
    )

# Allow the gate to be overridden at run time without editing the source.
min_gain: float = float(min_gain_widget) if min_gain_widget else MIN_GAIN

# Fully qualified UC model name (three-part namespace: catalog.schema.object).
full_model_name: str = f"{catalog}.{schema}.{model_name}"

# Resolve the experiment path. Canonical convention: /Users/{user}/yolo-cv.
current_user: str = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
experiment_path: str = experiment_path_widget or f"/Users/{current_user}/yolo-cv"

# COMMAND ----------

# -----------------------------------------------------------------------------
# MLflow client bound to the Unity Catalog model registry.
# -----------------------------------------------------------------------------
mlflow.set_registry_uri("databricks-uc")
client: MlflowClient = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------


def get_experiment_id(mlflow_client: MlflowClient, path: str) -> str:
    """Return the MLflow experiment id for ``path``.

    Args:
        mlflow_client: An ``MlflowClient`` instance.
        path: The workspace path of the experiment (e.g. ``/Users/{user}/yolo-cv``).

    Returns:
        The experiment id as a string.

    Raises:
        ValueError: If no experiment exists at ``path``.
    """
    experiment = mlflow_client.get_experiment_by_name(path)
    if experiment is None:
        raise ValueError(f"No MLflow experiment found at path '{path}'.")
    return experiment.experiment_id


def list_runs_by_metric(
    mlflow_client: MlflowClient,
    experiment_id: str,
    metric: str,
) -> list[Run]:
    """List finished runs of an experiment ordered by ``metric`` descending.

    Args:
        mlflow_client: An ``MlflowClient`` instance.
        experiment_id: The id of the experiment to search.
        metric: The metric name to order by (highest first).

    Returns:
        A list of :class:`mlflow.entities.Run`, best (highest metric) first.
    """
    return mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
    )


def get_metric(run: Run, metric: str) -> Optional[float]:
    """Safely read a numeric metric from a run.

    Args:
        run: The MLflow run.
        metric: The metric name.

    Returns:
        The metric value as a float, or ``None`` when the run did not log it.
    """
    value = run.data.metrics.get(metric)
    return float(value) if value is not None else None


def find_version_for_run(
    mlflow_client: MlflowClient,
    name: str,
    run_id: str,
) -> Optional[ModelVersion]:
    """Find the registered model version produced by a given training run.

    Args:
        mlflow_client: An ``MlflowClient`` instance.
        name: Fully qualified UC model name (``catalog.schema.object``).
        run_id: The MLflow run id that produced the version.

    Returns:
        The matching :class:`ModelVersion`, or ``None`` when no version is
        linked to ``run_id``.
    """
    matches = mlflow_client.search_model_versions(
        f"name = '{name}' AND run_id = '{run_id}'"
    )
    if not matches:
        return None
    # A single run should map to one version; pick the newest if several exist.
    return max(matches, key=lambda mv: int(mv.version))


def get_champion_map50(
    mlflow_client: MlflowClient,
    name: str,
    metric: str,
) -> Optional[float]:
    """Read the gating metric of the current ``@champion``, if any.

    Handles the first-deploy case gracefully: when no ``@champion`` alias exists
    yet, ``None`` is returned so the caller can promote unconditionally.

    Args:
        mlflow_client: An ``MlflowClient`` instance.
        name: Fully qualified UC model name (``catalog.schema.object``).
        metric: The metric name used for the gate.

    Returns:
        The champion's metric value, or ``None`` when there is no champion or the
        champion's run did not log the metric.
    """
    try:
        champion = mlflow_client.get_model_version_by_alias(name, "champion")
    except Exception:
        # No '@champion' alias yet (first deploy) or it is unresolvable.
        return None

    if champion.run_id is None:
        return None
    champion_run = mlflow_client.get_run(champion.run_id)
    return get_metric(champion_run, metric)


# COMMAND ----------


def promote_models(
    mlflow_client: MlflowClient,
    name: str,
    experiment: str,
    metric: str,
    gain_gate: float,
) -> dict[str, object]:
    """Run the promotion gate and assign UC aliases.

    The best candidate becomes ``@champion`` only when its metric beats the
    incumbent champion by at least ``gain_gate`` (relative). The second-best
    registered version is always assigned to ``@challenger``. The decision is
    recorded as a tag (``promotion_decision``) on the winning run.

    Args:
        mlflow_client: An ``MlflowClient`` bound to the UC registry.
        name: Fully qualified UC model name (``catalog.schema.object``).
        experiment: Workspace path of the MLflow experiment.
        metric: Metric used to rank runs and gate promotion (e.g. ``mAP50``).
        gain_gate: Minimum relative gain to promote (e.g. ``1.02`` == +2%).

    Returns:
        A dictionary describing the outcome: the chosen versions, the metric
        values compared, and the textual decision.

    Raises:
        ValueError: If no usable run/version can be found for promotion.
    """
    experiment_id = get_experiment_id(mlflow_client, experiment)
    runs = list_runs_by_metric(mlflow_client, experiment_id, metric)
    if not runs:
        raise ValueError(
            f"No FINISHED runs found in experiment '{experiment}' to promote."
        )

    # ---- Identify the best run and its registered version. -------------------
    best_run = runs[0]
    best_map50 = get_metric(best_run, metric)
    if best_map50 is None:
        raise ValueError(
            f"Best run '{best_run.info.run_id}' has no '{metric}' metric logged."
        )

    best_version = find_version_for_run(mlflow_client, name, best_run.info.run_id)
    if best_version is None:
        raise ValueError(
            f"No registered version of '{name}' is linked to best run "
            f"'{best_run.info.run_id}'. Ensure 03_train registered the model."
        )

    # ---- Read the incumbent champion's metric (None on first deploy). --------
    champion_map50 = get_champion_map50(mlflow_client, name, metric)

    # ---- Apply the relative-gain promotion gate. -----------------------------
    if champion_map50 is None:
        # First deploy: nothing to compare against, promote unconditionally.
        promote = True
        decision = "promoted"
        reason = "first_deploy_no_champion"
    elif best_map50 > champion_map50 * gain_gate:
        promote = True
        decision = "promoted"
        reason = (
            f"{metric} {best_map50:.5f} > champion {champion_map50:.5f} "
            f"* {gain_gate} ({champion_map50 * gain_gate:.5f})"
        )
    else:
        promote = False
        decision = "kept_champion"
        reason = (
            f"{metric} {best_map50:.5f} did not beat champion "
            f"{champion_map50:.5f} * {gain_gate} ({champion_map50 * gain_gate:.5f})"
        )

    if promote:
        mlflow_client.set_registered_model_alias(name, "champion", best_version.version)

    # ---- Assign the second-best registered version to '@challenger'. ---------
    challenger_version: Optional[str] = None
    for run in runs[1:]:
        candidate = find_version_for_run(mlflow_client, name, run.info.run_id)
        if candidate is not None and candidate.version != best_version.version:
            challenger_version = candidate.version
            break

    if challenger_version is not None:
        mlflow_client.set_registered_model_alias(
            name, "challenger", challenger_version
        )

    # ---- Audit trail: tag the winning run with the decision. -----------------
    mlflow_client.set_tag(best_run.info.run_id, "promotion_decision", decision)

    return {
        "model_name": name,
        "best_run_id": best_run.info.run_id,
        "best_version": best_version.version,
        f"best_{metric}": best_map50,
        f"champion_{metric}": champion_map50,
        "gain_gate": gain_gate,
        "challenger_version": challenger_version,
        "decision": decision,
        "reason": reason,
    }


# COMMAND ----------

# -----------------------------------------------------------------------------
# Entry point: run the promotion gate and surface the result.
# -----------------------------------------------------------------------------
result = promote_models(
    mlflow_client=client,
    name=full_model_name,
    experiment=experiment_path,
    metric=RANKING_METRIC,
    gain_gate=min_gain,
)

print("Promotion summary")
print("-----------------")
for key, value in result.items():
    print(f"  {key}: {value}")

# Return a JSON-serializable summary so an orchestrating Databricks Job can
# branch on the decision downstream.
dbutils.notebook.exit(str(result))
