from unittest import mock

import pytest

from mlflow.tracing.archival import (
    disable_databricks_trace_archival,
    enable_databricks_trace_archival,
)
from mlflow.version import IS_TRACING_SDK_ONLY

if IS_TRACING_SDK_ONLY:
    pytest.skip("Databricks archival enablement requires skinny", allow_module_level=True)


def test_enable_databricks_trace_archival_import_error():
    with mock.patch.dict("sys.modules", {"databricks.agents.archive": None}):
        with pytest.raises(ImportError, match="databricks-agents"):
            enable_databricks_trace_archival(
                experiment_id="123", delta_table_fullname="catalog.schema.table"
            )


def test_disable_databricks_trace_archival_import_error():
    with mock.patch.dict("sys.modules", {"databricks.agents.archive": None}):
        with pytest.raises(ImportError, match="databricks-agents"):
            disable_databricks_trace_archival(experiment_id="123")


def test_enable_databricks_trace_archival_with_explicit_experiment_id():
    mock_enable = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.agents.archive": mock.MagicMock(enable_trace_archival=mock_enable)},
    ):
        enable_databricks_trace_archival(
            experiment_id="123", delta_table_fullname="catalog.schema.table"
        )
        mock_enable.assert_called_once_with(
            experiment_id="123", table_fullname="catalog.schema.table"
        )


def test_enable_databricks_trace_archival_with_default_experiment_id():
    mock_enable = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.agents.archive": mock.MagicMock(enable_trace_archival=mock_enable)},
    ):
        with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="default_exp"):
            enable_databricks_trace_archival(delta_table_fullname="catalog.schema.table")
            mock_enable.assert_called_once_with(
                experiment_id="default_exp", table_fullname="catalog.schema.table"
            )


def test_disable_databricks_trace_archival_with_explicit_experiment_id():
    mock_disable = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.agents.archive": mock.MagicMock(disable_trace_archival=mock_disable)},
    ):
        disable_databricks_trace_archival(experiment_id="123")
        mock_disable.assert_called_once_with(experiment_id="123")


def test_disable_databricks_trace_archival_with_default_experiment_id():
    mock_disable = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"databricks.agents.archive": mock.MagicMock(disable_trace_archival=mock_disable)},
    ):
        with mock.patch("mlflow.tracking.fluent._get_experiment_id", return_value="default_exp"):
            disable_databricks_trace_archival()
            mock_disable.assert_called_once_with(experiment_id="default_exp")
