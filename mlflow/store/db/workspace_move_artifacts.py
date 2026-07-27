from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa

from mlflow.store.tracking.dbmodels.models import SqlExperiment
from mlflow.store.workspace.dbmodels.models import SqlWorkspace
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.workspace_utils import WORKSPACES_DIR_NAME


@dataclass(frozen=True)
class ExperimentRetargetPlan:
    """Retarget plan for one experiment's artifact root."""

    experiment_id: int
    experiment_name: str
    old_root: str
    new_root: str


@dataclass(frozen=True)
class SkippedRetarget:
    """Experiment left on its current artifact root, with the reason."""

    experiment_name: str
    artifact_location: str
    reason: str


def _normalized(uri: str) -> str:
    return uri.rstrip("/")


def _resolve_workspace_root_base(conn, workspace: str, default_artifact_root: str | None) -> str:
    """Mirror the workspace provider's artifact root resolution for a workspace.

    A workspace-level default_artifact_root is used as is, matching the provider
    returning append_workspace_prefix=False for it. The server-level root gets the
    workspaces/<name> suffix appended, matching the provider default.
    """
    workspaces = SqlWorkspace.__table__
    workspace_root = conn.execute(
        sa.select(workspaces.c.default_artifact_root).where(workspaces.c.name == workspace)
    ).scalar()
    if workspace_root:
        return workspace_root
    if not default_artifact_root:
        raise RuntimeError(
            f"Cannot determine the artifact root for workspace {workspace!r}: the "
            "workspace has no default_artifact_root configured. Pass --default-artifact-root "
            "with the same value the tracking server is started with."
        )
    return append_to_uri_path(default_artifact_root, WORKSPACES_DIR_NAME, workspace)


def build_experiment_retarget_plans(
    conn,
    experiment_names: list[str],
    source_workspace: str,
    target_workspace: str,
    default_artifact_root: str,
) -> tuple[list[ExperimentRetargetPlan], list[SkippedRetarget]]:
    """Plan the artifact root retarget for the experiments being moved.

    Only experiments on a server-derived layout are retargeted to the artifact
    root resolved for the target workspace: either the pre-workspace
    <default_artifact_root>/<experiment_id> layout or the layout derived for the
    source workspace. Everything else is reported and left unchanged, since a
    custom location cannot be assumed to follow any layout.
    """
    experiments = SqlExperiment.__table__
    target_base = _resolve_workspace_root_base(conn, target_workspace, default_artifact_root)
    source_base = _resolve_workspace_root_base(conn, source_workspace, default_artifact_root)

    rows = conn.execute(
        sa
        .select(
            experiments.c.experiment_id,
            experiments.c.name,
            experiments.c.artifact_location,
        )
        .where(
            experiments.c.workspace == source_workspace,
            experiments.c.name.in_(experiment_names),
        )
        .order_by(experiments.c.name)
    ).fetchall()

    plans: list[ExperimentRetargetPlan] = []
    skipped: list[SkippedRetarget] = []
    for experiment_id, name, current_root in rows:
        recognized_roots = {
            _normalized(append_to_uri_path(default_artifact_root, str(experiment_id))),
            _normalized(append_to_uri_path(source_base, str(experiment_id))),
        }
        if not current_root or _normalized(current_root) not in recognized_roots:
            skipped.append(
                SkippedRetarget(
                    experiment_name=name,
                    artifact_location=current_root or "",
                    reason="not on a layout derived from the default artifact root",
                )
            )
            continue
        new_root = append_to_uri_path(target_base, str(experiment_id))
        if _normalized(new_root) == _normalized(current_root):
            skipped.append(
                SkippedRetarget(
                    experiment_name=name,
                    artifact_location=current_root,
                    reason="already at the target artifact root",
                )
            )
            continue
        plans.append(
            ExperimentRetargetPlan(
                experiment_id=experiment_id,
                experiment_name=name,
                old_root=current_root,
                new_root=new_root,
            )
        )
    return plans, skipped


def apply_experiment_retargets(conn, plans: tuple[ExperimentRetargetPlan, ...]) -> None:
    """Rewrite experiments.artifact_location to the planned workspace-aware roots.

    Runs inside the caller's move transaction so the workspace flip and the
    retarget commit atomically. Only the experiment rows change. Stored run,
    logged model and trace URIs are absolute and keep resolving as they are.
    """
    if not plans:
        return
    experiments = SqlExperiment.__table__
    conn.execute(
        experiments
        .update()
        .where(experiments.c.experiment_id == sa.bindparam("pk"))
        .values(artifact_location=sa.bindparam("new_root")),
        [{"pk": plan.experiment_id, "new_root": plan.new_root} for plan in plans],
    )
