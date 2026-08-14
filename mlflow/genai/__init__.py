import importlib.util
import warnings

import numpy as np


def _warn_if_databricks_connect_downgraded_numpy() -> None:
    """
    Warn when `databricks-connect` (pulled in by `databricks-agents`, a dependency of
    `mlflow.genai.datasets` on Databricks) appears to have downgraded numpy to <2 in
    this environment, which can cause cryptic errors such as
    `AttributeError: module 'numpy' has no attribute 'long'` in packages that require
    `numpy>=2` (see #24690).
    """
    try:
        numpy_major_version = int(np.__version__.split(".")[0])
    except (ValueError, IndexError):
        return

    if numpy_major_version >= 2:
        return

    # find_spec raises ModuleNotFoundError, rather than returning None, when the
    # "databricks" parent package isn't installed at all.
    try:
        databricks_connect_installed = importlib.util.find_spec("databricks.connect") is not None
    except (ImportError, ValueError):
        return

    if not databricks_connect_installed:
        return

    warnings.warn(
        f"Detected numpy {np.__version__} alongside `databricks-connect`, which pins "
        "`numpy<2`. If this environment previously had `numpy>=2`, installing "
        "`databricks-agents` (a dependency of `mlflow.genai.datasets` on Databricks) "
        "likely downgraded numpy, which can break other packages that require "
        "`numpy>=2`. Consider reinstalling `numpy>=2` or using a dedicated "
        "environment for `databricks-agents`.",
        stacklevel=2,
    )


_warn_if_databricks_connect_downgraded_numpy()

from mlflow.genai import (
    datasets,
    judges,
    scorers,
)
from mlflow.genai.agent_tester import test_agent
from mlflow.genai.datasets import (
    EvaluationDatasetVersion,
    create_dataset,
    delete_dataset,
    delete_dataset_tag,
    get_dataset,
    search_datasets,
    set_dataset_tags,
)
from mlflow.genai.evaluation import evaluate, to_predict_fn
from mlflow.genai.git_versioning import disable_git_model_versioning, enable_git_model_versioning
from mlflow.genai.judges import make_judge
from mlflow.genai.labeling import (
    Agent,
    LabelingSession,
    ReviewApp,
    create_labeling_session,
    delete_labeling_session,
    get_labeling_session,
    get_labeling_sessions,
    get_review_app,
)
from mlflow.genai.mcp_servers import (
    create_mcp_access_endpoint,
    delete_mcp_access_endpoint,
    delete_mcp_server,
    delete_mcp_server_alias,
    delete_mcp_server_tag,
    delete_mcp_server_version,
    delete_mcp_server_version_tag,
    get_latest_mcp_server_version,
    get_mcp_access_endpoint,
    get_mcp_server,
    get_mcp_server_version,
    get_mcp_server_version_by_alias,
    refresh_mcp_server_version_tools,
    register_mcp_server,
    register_mcp_server_from_url,
    search_mcp_access_endpoints,
    search_mcp_server_versions,
    search_mcp_servers,
    set_mcp_server_alias,
    set_mcp_server_tag,
    set_mcp_server_version_tag,
    update_mcp_access_endpoint,
    update_mcp_server,
    update_mcp_server_version,
)
from mlflow.genai.optimize import optimize_prompt, optimize_prompts
from mlflow.genai.prompts import (
    delete_prompt_alias,
    delete_prompt_model_config,
    delete_prompt_tag,
    delete_prompt_version_tag,
    get_prompt_tags,
    load_prompt,
    register_prompt,
    search_prompts,
    set_prompt_alias,
    set_prompt_model_config,
    set_prompt_tag,
    set_prompt_version_tag,
)
from mlflow.genai.scheduled_scorers import (
    ScorerScheduleConfig,
)
from mlflow.genai.scorers import (
    Scorer,
    delete_scorer,
    get_scorer,
    list_scorers,
    make_scorer_ensemble,
    scorer,
)
from mlflow.genai.simulators import ConversationSimulator

__all__ = [
    "datasets",
    "test_agent",
    "evaluate",
    "to_predict_fn",
    "Scorer",
    "scorer",
    "make_scorer_ensemble",
    "get_scorer",
    "list_scorers",
    "delete_scorer",
    "judges",
    "make_judge",
    "scorers",
    "EvaluationDatasetVersion",
    "create_dataset",
    "delete_dataset",
    "delete_dataset_tag",
    "get_dataset",
    "search_datasets",
    "set_dataset_tags",
    "load_prompt",
    "register_prompt",
    "search_prompts",
    "delete_prompt_alias",
    "set_prompt_alias",
    "optimize_prompts",
    "optimize_prompt",
    "get_prompt_tags",
    "set_prompt_tag",
    "set_prompt_version_tag",
    "delete_prompt_tag",
    "delete_prompt_version_tag",
    "set_prompt_model_config",
    "delete_prompt_model_config",
    "ScorerScheduleConfig",
    "Agent",
    "LabelingSession",
    "ReviewApp",
    "get_review_app",
    "create_labeling_session",
    "get_labeling_sessions",
    "get_labeling_session",
    "delete_labeling_session",
    # git model versioning
    "disable_git_model_versioning",
    "enable_git_model_versioning",
    # conversation simulation
    "ConversationSimulator",
    # MCP server registry
    "register_mcp_server",
    "register_mcp_server_from_url",
    "get_mcp_server",
    "search_mcp_servers",
    "update_mcp_server",
    "delete_mcp_server",
    "get_mcp_server_version",
    "get_mcp_server_version_by_alias",
    "get_latest_mcp_server_version",
    "search_mcp_server_versions",
    "update_mcp_server_version",
    "refresh_mcp_server_version_tools",
    "delete_mcp_server_version",
    "create_mcp_access_endpoint",
    "get_mcp_access_endpoint",
    "search_mcp_access_endpoints",
    "update_mcp_access_endpoint",
    "delete_mcp_access_endpoint",
    "set_mcp_server_tag",
    "delete_mcp_server_tag",
    "set_mcp_server_version_tag",
    "delete_mcp_server_version_tag",
    "set_mcp_server_alias",
    "delete_mcp_server_alias",
]
