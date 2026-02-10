from collections.abc import Generator
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from clint.config import Config
from clint.linter import Violation, lint_file
from clint.rules.workspace_query_isolation import (
    WorkspaceQueryIsolation,
    _get_workspace_isolated_models,
    _get_workspace_overrides,
    _parse_workspace_store,
)

BASE_STORE_PATH = Path("mlflow/store/tracking/sqlalchemy_store.py")

# Simulated workspace overrides (methods the workspace store overrides).
MOCK_WORKSPACE_OVERRIDES = frozenset(
    {
        "__init__",
        "_get_query",
        "_trace_query",
        "_validate_run_accessible",
        "_validate_trace_accessible",
        "_validate_dataset_accessible",
        "_create_default_experiment",
        "_initialize_store_state",
    }
)

# Simulated workspace-isolated models (models handled in _get_query).
MOCK_WORKSPACE_MODELS = frozenset(
    {
        "SqlExperiment",
        "SqlRun",
        "SqlTraceInfo",
        "SqlLoggedModel",
    }
)


@pytest.fixture(autouse=True)
def _mock_workspace_discovery() -> Generator[None, None, None]:
    _get_workspace_overrides.cache_clear()
    _get_workspace_isolated_models.cache_clear()
    _parse_workspace_store.cache_clear()
    with (
        patch(
            "clint.rules.workspace_query_isolation._get_workspace_overrides",
            return_value=MOCK_WORKSPACE_OVERRIDES,
        ),
        patch(
            "clint.rules.workspace_query_isolation._get_workspace_isolated_models",
            return_value=MOCK_WORKSPACE_MODELS,
        ),
    ):
        yield
    _get_workspace_overrides.cache_clear()
    _get_workspace_isolated_models.cache_clear()
    _parse_workspace_store.cache_clear()


def _lint(code: str, index_path: Path, path: Path = BASE_STORE_PATH) -> list[Violation]:
    config = Config(select={WorkspaceQueryIsolation.name})
    return lint_file(path, dedent(code).strip() + "\n", config, index_path)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            """
class Store:
    def search_runs(self, session):
        session.query(SqlRun).filter(SqlRun.lifecycle_stage == "deleted").all()
""",
            id="public_method_session_query",
        ),
        pytest.param(
            """
class Store:
    def _helper(self, session):
        session.query(SqlRun).filter(SqlRun.run_uuid == "abc").one_or_none()
""",
            id="private_method_session_query",
        ),
        pytest.param(
            """
class Store:
    def get_run(self, session, run_id):
        session.query(SqlRun).get(run_id)
""",
            id="get_call",
        ),
        pytest.param(
            """
class Store:
    def get_run(self, session, run_id):
        session.query(SqlRun).filter(SqlRun.run_uuid == run_id).one_or_none()
""",
            id="id_filter_no_workspace",
        ),
        pytest.param(
            """
class Store:
    def get_run(self, session, run_id):
        session.query(SqlRun).filter_by(run_uuid=run_id).one_or_none()
""",
            id="filter_by_id_no_workspace",
        ),
        pytest.param(
            """
class Store:
    def get_runs(self, session, run_ids):
        session.query(SqlRun).filter(SqlRun.run_uuid.in_(run_ids)).all()
""",
            id="in_filter_no_workspace",
        ),
        pytest.param(
            """
class Store:
    def get_experiment(self, session, experiment_id):
        session.query(SqlExperiment).filter(
            SqlExperiment.experiment_id == experiment_id
        ).one()
""",
            id="workspace_model_direct_query",
        ),
    ],
)
def test_flags_session_query(code: str, index_path: Path) -> None:
    violations = _lint(code, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, WorkspaceQueryIsolation)


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            """
class Store:
    def _get_query(self, session, model):
        return session.query(model)
""",
            id="workspace_override_method",
        ),
        pytest.param(
            """
class Store:
    def _trace_query(self, session):
        return session.query(SqlTraceInfo).all()
""",
            id="another_workspace_override",
        ),
        pytest.param(
            """
class Store:
    def get_run(self, session, run_id):
        return self._get_query(session, SqlRun).filter(SqlRun.run_uuid == run_id).one()
""",
            id="uses_get_query_helper",
        ),
        pytest.param(
            """
class Store:
    def delete_tag(self, session, run_id, key):
        self._validate_run_accessible(session, run_id)
        session.query(SqlRun).filter(SqlRun.run_uuid == run_id).delete()
""",
            id="calls_workspace_helper_then_session_query",
        ),
        pytest.param(
            """
class Store:
    def get_trace(self, session, trace_id):
        trace = self._trace_query(session).filter(SqlTraceInfo.request_id == trace_id).one()
        session.query(SqlTraceInfo).filter(SqlTraceInfo.request_id == trace_id).all()
""",
            id="calls_workspace_override_then_session_query",
        ),
        pytest.param(
            """
class Store:
    def _initialize_store_state(self, session):
        session.query(SqlExperiment).filter(SqlExperiment.workspace != "default").first()
""",
            id="allowlisted_method",
        ),
        pytest.param(
            """
class Store:
    def check(self, session):
        session.query(SqlRun).count()  # clint: disable=workspace-query-isolation
""",
            id="clint_disable_comment",
        ),
        pytest.param(
            """
class Store:
    def helper(self, session):
        session.execute(text("SELECT 1"))
""",
            id="non_query_call",
        ),
        pytest.param(
            """
class Store:
    def get_experiment(self, session, experiment_id):
        return self._get_query(session, SqlExperiment).filter(
            SqlExperiment.experiment_id == experiment_id
        ).one()
""",
            id="get_query_chain",
        ),
        pytest.param(
            """
class Store:
    def get_metric(self, session, run_id, key):
        self._validate_run_accessible(session, run_id)
        return session.query(SqlRun).filter(
            SqlRun.run_uuid == run_id
        ).first()
""",
            id="workspace_helper_protects_workspace_model",
        ),
        pytest.param(
            """
class Store:
    def get_tag(self, session, run_id, key):
        return session.query(SqlTag).filter(
            SqlTag.run_uuid == run_id, SqlTag.key == key
        ).first()
""",
            id="child_model_implicitly_safe",
        ),
        pytest.param(
            """
class Store:
    def get_metrics(self, session, run_id):
        return session.query(SqlMetric).filter(
            SqlMetric.run_uuid == run_id
        ).all()
""",
            id="non_workspace_model_not_flagged",
        ),
    ],
)
def test_no_violation(code: str, index_path: Path) -> None:
    violations = _lint(code, index_path)
    assert len(violations) == 0


def test_ignores_non_base_store_path(index_path: Path) -> None:
    code = """
class Store:
    def search_runs(self, session):
        session.query(SqlRun).filter(SqlRun.lifecycle_stage == "deleted").all()
"""
    violations = _lint(
        code,
        index_path,
        path=Path("mlflow/store/tracking/sqlalchemy_workspace_store.py"),
    )
    assert len(violations) == 0


def test_inner_function_inherits_method_safety(index_path: Path) -> None:
    code = """
class Store:
    def delete_tag(self, session, run_id, key):
        self._validate_run_accessible(session, run_id)
        def _do_delete():
            session.query(SqlRun).filter(SqlRun.run_uuid == run_id).delete()
        _do_delete()
"""
    violations = _lint(code, index_path)
    assert len(violations) == 0


def test_inner_function_without_helper_flagged(index_path: Path) -> None:
    code = """
class Store:
    def get_runs(self, session):
        def _fetch():
            return session.query(SqlRun).all()
        return _fetch()
"""
    violations = _lint(code, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, WorkspaceQueryIsolation)
