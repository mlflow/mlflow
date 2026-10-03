"""
Per-tool authorization for the MCP endpoint served by the tracking server.

Every tool served over ``/mcp`` maps to how its resource is resolved from the tool arguments
and the permission the caller needs on it. Each rule mirrors the validator of the corresponding
tracking operation in ``BEFORE_REQUEST_HANDLERS``, so a call through MCP is authorized exactly
like the same operation through the REST API; where the REST gate differs from the tool verb,
the entry says which handler it mirrors.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from mlflow.exceptions import MlflowException
from mlflow.mcp.request_context import get_mcp_request_username
from mlflow.mcp.server_app import McpToolPolicy
from mlflow.mcp.tools._args import as_list, as_tag_dict, as_view_type, check_non_negative
from mlflow.mcp.tools._types import (
    ExperimentInfo,
    ExperimentPage,
    ExperimentRef,
    RegisteredScorer,
    ScorerList,
)
from mlflow.mcp.tools.scorers import list_scorers
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.server import auth as auth_module
from mlflow.server.auth.permissions import Permission
from mlflow.server.handlers import _get_tracking_store
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

PermissionName = Literal["can_read", "can_update", "can_delete", "can_manage"]
Resolver = Callable[[dict[str, Any], str], list[Permission]]
Check = Callable[[dict[str, Any], str], bool]


@dataclass(frozen=True)
class McpToolRule:
    """
    Authorization rule for one served MCP tool.

    Args:
        resolve: Maps the tool arguments and the caller to the caller's permissions on every
            resource the call touches. An empty list denies (missing argument or resource).
        permission: Attribute every resolved permission must grant.
        check: For tools whose REST validator is not a single resource plus permission
            (create gates, composites, filtered searches): decides directly.
    """

    resolve: Resolver | None = None
    permission: PermissionName | None = None
    check: Check | None = None

    def __post_init__(self):
        if (self.check is None) == (self.resolve is None or self.permission is None):
            raise ValueError("Specify either `resolve` and `permission`, or `check`")

    def allows(self, arguments: dict[str, Any], username: str) -> bool:
        if self.check is not None:
            return self.check(arguments, username)
        permissions = self.resolve(arguments, username)
        return bool(permissions) and all(getattr(p, self.permission) for p in permissions)


def _experiment_permission(experiment_id: Any, username: str) -> list[Permission]:
    if experiment_id is None:
        return []
    return [auth_module._get_experiment_permission(str(experiment_id), username)]


def _experiment(arguments: dict[str, Any], username: str) -> list[Permission]:
    return _experiment_permission(arguments.get("experiment_id"), username)


def _experiment_by_id_or_name(arguments: dict[str, Any], username: str) -> list[Permission]:
    if arguments.get("experiment_id") is not None:
        return _experiment(arguments, username)
    if not (name := arguments.get("experiment_name")):
        return []
    experiment = _get_tracking_store().get_experiment_by_name(name)
    return [] if experiment is None else _experiment_permission(experiment.experiment_id, username)


def _run(arguments: dict[str, Any], username: str) -> list[Permission]:
    if (run_id := arguments.get("run_id")) is None:
        return []
    try:
        return [auth_module._get_run_permission(str(run_id), username)]
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return []
        raise


def _trace(arguments: dict[str, Any], username: str) -> list[Permission]:
    if (trace_id := arguments.get("trace_id")) is None:
        return []
    # Resolves the trace's experiment; a missing trace yields NO_PERMISSIONS.
    return [auth_module._get_permission_from_trace(str(trace_id), username)]


def _can_create_experiment(arguments: dict[str, Any], username: str) -> bool:
    return auth_module._can_create_in_workspace(username)


def _can_create_run(arguments: dict[str, Any], username: str) -> bool:
    # CreateRun -> validate_can_update_experiment. Given only a name, the CLI creates a missing
    # experiment on the fly, which is gated like CreateExperiment -> validate_can_create_experiment.
    # The run is nested under the parent named by ``parent_run_id`` or by the ``mlflow.parentRunId``
    # tag the tool passes through, so that run must be readable (and exist): otherwise a caller
    # could attach runs under, or probe for, runs it cannot see. The tags are parsed as the tool
    # parses them, so both ways of naming the parent are checked alike. REST CreateRun does not
    # check the parent named by the tag.
    parent_run_id = arguments.get("parent_run_id")
    tag_parent_run_id = as_tag_dict(arguments.get("tags")).get(MLFLOW_PARENT_RUN_ID)
    if parent_run_id is not None and tag_parent_run_id is not None:
        if parent_run_id != tag_parent_run_id:
            return False
    parent = tag_parent_run_id if parent_run_id is None else parent_run_id
    if parent is not None:
        parent_permissions = _run({"run_id": parent}, username) if parent else []
        if not parent_permissions or not parent_permissions[0].can_read:
            return False
    if arguments.get("experiment_id") is not None:
        return all(p.can_update for p in _experiment(arguments, username))
    if not (name := arguments.get("experiment_name")):
        return False
    experiment = _get_tracking_store().get_experiment_by_name(name)
    if experiment is None:
        return auth_module._can_create_in_workspace(username)
    return auth_module._get_experiment_permission(experiment.experiment_id, username).can_update


def _can_link_traces_to_run(arguments: dict[str, Any], username: str) -> bool:
    # LinkTracesToRun -> validate_can_link_traces_to_run: update on the run's experiment and
    # read on every trace's experiment.
    run_permissions = _run(arguments, username)
    if not run_permissions or not run_permissions[0].can_update:
        return False
    trace_ids = arguments.get("trace_ids") or []
    return bool(trace_ids) and all(
        auth_module._get_permission_from_trace(str(trace_id), username).can_read
        for trace_id in trace_ids
    )


def _can_list_scorers(arguments: dict[str, Any], username: str) -> bool:
    # ListScorers -> validate_can_read_scorer_list: read on the experiment when one is given.
    # Without one the CLI only lists the built-in catalog, which is not a store resource.
    if (experiment_id := arguments.get("experiment_id")) is None:
        return True
    return auth_module._get_experiment_permission(str(experiment_id), username).can_read


def _scorer_exists(experiment_id: str, name: str) -> bool:
    try:
        _get_tracking_store().get_scorer(experiment_id, name)
    except MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return False
        raise
    return True


def _can_register_scorer(arguments: dict[str, Any], username: str) -> bool:
    # RegisterScorer -> validate_can_update_experiment. Registering an existing name adds a
    # version to that scorer, which over REST an experiment editor may do (and is then granted
    # MANAGE on it). Here a new version is gated like the scorer-level validators
    # (validate_can_update_scorer), so only a caller who may update the scorer can replace it;
    # a genuinely new scorer needs update on the experiment. Both denials are the generic one.
    experiment_id, name = arguments.get("experiment_id"), arguments.get("name")
    if experiment_id is None or not name:
        return False
    experiment_id = str(experiment_id)
    if _scorer_exists(experiment_id, name):
        return auth_module._get_scorer_permission(experiment_id, name, username).can_update
    return auth_module._get_experiment_permission(experiment_id, username).can_update


def _any_authenticated(arguments: dict[str, Any], username: str) -> bool:
    return True


MCP_TOOL_RULES: dict[str, McpToolRule] = {
    # Experiments
    "create_experiment": McpToolRule(check=_can_create_experiment),
    "get_experiment": McpToolRule(_experiment_by_id_or_name, "can_read"),
    # Results are filtered per caller by ``search_readable_experiments`` (SearchExperiments ->
    # filter_search_experiments).
    "search_experiments": McpToolRule(check=_any_authenticated),
    "rename_experiment": McpToolRule(_experiment, "can_update"),
    "update_experiment": McpToolRule(_experiment, "can_update"),
    "delete_experiment": McpToolRule(_experiment, "can_delete"),
    # RestoreExperiment -> validate_can_delete_experiment
    "restore_experiment": McpToolRule(_experiment, "can_delete"),
    # Runs
    "create_run": McpToolRule(check=_can_create_run),
    "describe_run": McpToolRule(_run, "can_read"),
    "list_runs": McpToolRule(_experiment, "can_read"),
    "delete_run": McpToolRule(_run, "can_delete"),
    # RestoreRun -> validate_can_delete_run
    "restore_run": McpToolRule(_run, "can_delete"),
    # Traces
    "get_trace": McpToolRule(_trace, "can_read"),
    "search_traces": McpToolRule(_experiment, "can_read"),
    "delete_traces": McpToolRule(_experiment, "can_delete"),
    "set_trace_tag": McpToolRule(_trace, "can_update"),
    # DeleteTraceTagV3 -> validate_can_update_trace_by_trace_id
    "delete_trace_tag": McpToolRule(_trace, "can_update"),
    "link_traces_to_run": McpToolRule(check=_can_link_traces_to_run),
    # Assessments: CreateAssessment / UpdateAssessment / DeleteAssessment all ->
    # validate_can_update_trace_by_trace_id
    "log_trace_feedback": McpToolRule(_trace, "can_update"),
    "log_trace_expectation": McpToolRule(_trace, "can_update"),
    "get_trace_assessment": McpToolRule(_trace, "can_read"),
    "update_trace_assessment": McpToolRule(_trace, "can_update"),
    "delete_trace_assessment": McpToolRule(_trace, "can_update"),
    # Scorers
    # Results are filtered per scorer by ``list_readable_scorers`` (ListScorers ->
    # filter_list_scorers).
    "list_scorers": McpToolRule(check=_can_list_scorers),
    "register_llm_judge_scorer": McpToolRule(check=_can_register_scorer),
}


def _permission_denied() -> MlflowException:
    return MlflowException("Permission denied", error_code=PERMISSION_DENIED)


def authorize_mcp_tool_call(
    tool_name: str, username: str | None, arguments: dict[str, Any]
) -> None:
    # Fail closed: no identity (the route is authenticated, so this is a wiring error) and no
    # rule both deny.
    if username is None:
        raise _permission_denied()
    rule = MCP_TOOL_RULES.get(tool_name)
    if rule is None or not rule.allows(arguments, username):
        raise _permission_denied()


def check_mcp_tool_coverage(tool_names: Iterable[str]) -> None:
    if missing := sorted(set(tool_names) - MCP_TOOL_RULES.keys()):
        raise MlflowException(
            f"MCP tools without an authorization rule: {missing}. Add an entry for each to "
            "MCP_TOOL_RULES in mlflow/server/auth/mcp_tools.py before serving it."
        )


def is_mcp_admin(username: str | None) -> bool:
    return username is not None and auth_module.store.get_user(username).is_admin


# Bounds the store round trips of one call when few experiments are readable. When the cap is hit
# before the page is filled, the rows collected so far are returned with a token to continue from.
SEARCH_READABLE_EXPERIMENTS_MAX_STORE_PAGES = 10


def search_readable_experiments(
    view: str = "active_only",
    max_results: int | None = SEARCH_MAX_RESULTS_DEFAULT,
    page_token: str | None = None,
    filter_string: str | None = None,
    order_by: list[str] | str | None = None,
) -> ExperimentPage:
    """
    ``search_experiments`` for a non-admin caller: same arguments and result, minus the
    experiments the caller cannot read.

    Like ``filter_search_experiments`` for the REST API, the store is paged further until
    ``max_results`` readable experiments are collected or the store is exhausted, so a page is
    not left short by unreadable rows. Each store request asks for exactly the remaining slots,
    so every row of a fetched page is consumed and the store's token always resumes right after
    it. At most ``SEARCH_READABLE_EXPERIMENTS_MAX_STORE_PAGES`` store pages are read per call;
    the result carries the last store token whenever the store has more rows.
    """
    if max_results is None:
        max_results = SEARCH_MAX_RESULTS_DEFAULT
    check_non_negative(max_results, "max_results")
    view_type = as_view_type(view)
    order_by_list = as_list(order_by)
    can_read = auth_module._role_based_read_predicate(get_mcp_request_username(), "experiment")
    client = MlflowClient()

    readable = []
    for _ in range(SEARCH_READABLE_EXPERIMENTS_MAX_STORE_PAGES):
        if (page_size := max_results - len(readable)) == 0:
            break
        page = client.search_experiments(
            view_type=view_type,
            max_results=page_size,
            filter_string=filter_string,
            order_by=order_by_list,
            page_token=page_token,
        )
        readable.extend(e for e in page if can_read(e.experiment_id))
        page_token = page.token or None
        if page_token is None:
            break

    return ExperimentPage(
        experiments=[ExperimentInfo.from_entity(e) for e in readable],
        next_page_token=page_token,
    )


def list_readable_scorers(
    experiment_id: str | None = None,
    builtin: bool = False,
    output: str | None = None,
) -> ScorerList:
    """
    ``list_scorers`` for a non-admin caller: the experiment's scorers the caller can read, with
    the same per-scorer check as ``filter_list_scorers`` for the REST API. The built-in catalog
    is not a store resource and is returned as is.
    """
    result = list_scorers(experiment_id=experiment_id, builtin=builtin, output=output)
    if experiment_id is None:
        return result
    can_read = auth_module.scorer_read_predicate(get_mcp_request_username())
    return ScorerList(scorers=[s for s in result.scorers if can_read(experiment_id, s.name)])


# Creator grants, mirroring the REST after-request handlers of the same operations
# (CreateExperiment -> set_can_manage_experiment_permission, RegisterScorer ->
# set_can_manage_scorer_permission). CreateRun grants nothing over REST; the experiment
# ``create_run`` creates from a missing name is granted through ``grant_created_experiment``, as
# CreateExperiment would, even when the run itself then fails.
def _grant_experiment_creator(username: str, result: ExperimentRef) -> None:
    auth_module.grant_creator_experiment_permission(username, result.experiment_id)


def _grant_scorer_creator(username: str, result: RegisteredScorer) -> None:
    # Only the creator of a scorer is granted MANAGE; a new version of an existing scorer
    # (which needs update on it) leaves the grants as they are.
    if result.version == 1:
        auth_module.grant_creator_scorer_permission(username, result.experiment_id, result.name)


def get_mcp_tool_policy() -> McpToolPolicy:
    return McpToolPolicy(
        authorize=authorize_mcp_tool_call,
        validate_coverage=check_mcp_tool_coverage,
        is_admin=is_mcp_admin,
        overrides={
            "search_experiments": search_readable_experiments,
            "list_scorers": list_readable_scorers,
        },
        on_success={
            "create_experiment": _grant_experiment_creator,
            "register_llm_judge_scorer": _grant_scorer_creator,
        },
        grant_created_experiment=auth_module.grant_creator_experiment_permission,
    )
