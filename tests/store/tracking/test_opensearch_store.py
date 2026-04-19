"""Unit tests for OpenSearchTrackingStore.

These tests mock the opensearch-py client to verify store logic in isolation,
without requiring a running OpenSearch cluster.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import (
    ExperimentTag,
    Metric,
    Param,
    RunTag,
    ViewType,
)
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.opensearch_mappings import (
    DEFAULT_INDEX_PREFIX,
    INDEX_MAPPINGS,
    get_all_index_configs,
    get_index_name,
)

# ---------------------------------------------------------------------------
# Mapping / helpers tests — no mocking needed
# ---------------------------------------------------------------------------


class TestOpenSearchMappings:
    def test_index_mappings_keys(self):
        expected = {
            "experiments",
            "runs",
            "metrics",
            "params",
            "tags",
            "traces",
            "trace_tags",
            "trace_metadata",
            "spans",
            "assessments",
        }
        assert set(INDEX_MAPPINGS.keys()) == expected

    def test_get_index_name(self):
        assert get_index_name("mlflow_", "experiments") == "mlflow_experiments"
        assert get_index_name("custom_", "runs") == "custom_runs"

    def test_get_all_index_configs(self):
        configs = get_all_index_configs("test_")
        assert "test_experiments" in configs
        assert "test_spans" in configs
        assert len(configs) == len(INDEX_MAPPINGS)

    def test_experiment_mapping_has_nested_tags(self):
        mapping = INDEX_MAPPINGS["experiments"]
        tags_mapping = mapping["mappings"]["properties"]["tags"]
        assert tags_mapping["type"] == "nested"
        assert "key" in tags_mapping["properties"]
        assert "value" in tags_mapping["properties"]

    def test_spans_mapping_has_full_text_content(self):
        mapping = INDEX_MAPPINGS["spans"]
        content_mapping = mapping["mappings"]["properties"]["content"]
        assert content_mapping["type"] == "text"

    def test_metrics_mapping_has_high_volume_settings(self):
        mapping = INDEX_MAPPINGS["metrics"]
        assert mapping["settings"]["number_of_shards"] == 3


# ---------------------------------------------------------------------------
# Query translator tests
# ---------------------------------------------------------------------------


class TestOpenSearchQueryTranslator:
    @pytest.fixture
    def translator(self):
        from mlflow.store.tracking.opensearch_query import OpenSearchQueryTranslator

        return OpenSearchQueryTranslator()

    def test_empty_filter(self, translator):
        query = translator.translate("", entity_type="run")
        assert "match_all" in str(query)

    def test_none_filter(self, translator):
        query = translator.translate(None, entity_type="run")
        assert "match_all" in str(query)


class TestBuildSortClause:
    def test_empty(self):
        from mlflow.store.tracking.opensearch_query import build_sort_clause

        assert build_sort_clause(None) == []
        assert build_sort_clause([]) == []

    def test_single_asc(self):
        from mlflow.store.tracking.opensearch_query import build_sort_clause

        result = build_sort_clause(["start_time ASC"])
        assert result == [{"start_time": {"order": "asc"}}]

    def test_single_desc(self):
        from mlflow.store.tracking.opensearch_query import build_sort_clause

        result = build_sort_clause(["start_time DESC"])
        assert result == [{"start_time": {"order": "desc"}}]

    def test_default_asc(self):
        from mlflow.store.tracking.opensearch_query import build_sort_clause

        result = build_sort_clause(["name"])
        assert result == [{"name": {"order": "asc"}}]

    def test_multiple(self):
        from mlflow.store.tracking.opensearch_query import build_sort_clause

        result = build_sort_clause(["start_time DESC", "name ASC"])
        assert len(result) == 2


class TestSqlLikeToWildcard:
    def test_percent_to_star(self):
        from mlflow.store.tracking.opensearch_query import _sql_like_to_wildcard

        assert _sql_like_to_wildcard("prod%") == "prod*"

    def test_underscore_to_question(self):
        from mlflow.store.tracking.opensearch_query import _sql_like_to_wildcard

        assert _sql_like_to_wildcard("v_1") == "v?1"

    def test_combined(self):
        from mlflow.store.tracking.opensearch_query import _sql_like_to_wildcard

        assert _sql_like_to_wildcard("%test_") == "*test?"


# ---------------------------------------------------------------------------
# Client manager tests
# ---------------------------------------------------------------------------


class TestOpenSearchClientManager:
    def test_parse_basic_uri(self):
        from mlflow.store.tracking._opensearch_client import OpenSearchClientManager

        mgr = OpenSearchClientManager("opensearch://localhost:9200")
        assert mgr._host == "localhost"
        assert mgr._port == 9200
        assert mgr._index_prefix == DEFAULT_INDEX_PREFIX

    def test_parse_uri_with_prefix(self):
        from mlflow.store.tracking._opensearch_client import OpenSearchClientManager

        mgr = OpenSearchClientManager("opensearch://localhost:9200/custom_prefix_")
        assert mgr._index_prefix == "custom_prefix_"

    def test_parse_https_uri(self):
        from mlflow.store.tracking._opensearch_client import OpenSearchClientManager

        mgr = OpenSearchClientManager("opensearch+https://user:pass@host:9200")
        assert mgr._host == "host"
        assert mgr._port == 9200
        assert mgr._kwargs.get("use_ssl") is True
        assert mgr._kwargs.get("http_auth") == ("user", "pass")

    def test_get_index_name(self):
        from mlflow.store.tracking._opensearch_client import OpenSearchClientManager

        mgr = OpenSearchClientManager("opensearch://localhost:9200")
        assert mgr.get_index_name("experiments") == f"{DEFAULT_INDEX_PREFIX}experiments"

    @patch("mlflow.store.tracking._opensearch_client.OpenSearchClientManager._parse_uri")
    def test_client_import_error(self, mock_parse):
        from mlflow.store.tracking._opensearch_client import OpenSearchClientManager

        mock_parse.return_value = ("localhost", 9200, "mlflow_", {})
        mgr = OpenSearchClientManager.__new__(OpenSearchClientManager)
        mgr._host = "localhost"
        mgr._port = 9200
        mgr._index_prefix = "mlflow_"
        mgr._kwargs = {}
        mgr._client = None
        mgr._store_uri = "opensearch://localhost:9200"

        with patch.dict("sys.modules", {"opensearchpy": None}):
            with pytest.raises(ImportError, match="opensearch-py is required"):
                _ = mgr.client


# ---------------------------------------------------------------------------
# OpenSearchTrackingStore tests (mocked client)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_os_client():
    """Create a mock OpenSearch client with minimal responses."""
    client = MagicMock()

    # Default responses for index existence check
    client.indices.exists.return_value = True

    # Default response for experiment get (default experiment)
    client.get.return_value = {
        "_id": "0",
        "_source": {
            "experiment_id": "0",
            "name": "Default",
            "artifact_location": "",
            "lifecycle_stage": "active",
            "creation_time": 1711089570000,
            "last_update_time": 1711089570000,
            "tags": [],
        },
    }

    # Default search response (includes aggregation for _get_latest_metrics)
    client.search.return_value = {
        "hits": {"hits": [], "total": {"value": 0}},
        "aggregations": {"max_id": {"value": 0}, "by_key": {"buckets": []}},
    }

    return client


@pytest.fixture
def store(mock_os_client):
    """Create an OpenSearchTrackingStore with a mocked client."""
    with patch(
        "mlflow.store.tracking._opensearch_client.OpenSearchClientManager.client",
        new_callable=lambda: property(lambda self: mock_os_client),
    ):
        from mlflow.store.tracking.opensearch_store import OpenSearchTrackingStore

        s = OpenSearchTrackingStore("opensearch://localhost:9200", artifact_uri="/tmp/artifacts")
        s._manager._client = mock_os_client
        return s


class TestExperimentCRUD:
    def test_create_experiment(self, store, mock_os_client):
        # Setup: no duplicate name found
        mock_os_client.search.return_value = {
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {"max_id": {"value": 1}},
        }
        mock_os_client.index.return_value = {"_id": "2", "result": "created"}

        exp_id = store.create_experiment("test-experiment")
        assert exp_id == "2"
        assert mock_os_client.index.called

    def test_create_experiment_duplicate_raises(self, store, mock_os_client):
        mock_os_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "experiment_id": "1",
                            "name": "dup",
                            "lifecycle_stage": "active",
                            "creation_time": 1000,
                            "last_update_time": 1000,
                            "tags": [],
                        }
                    }
                ],
                "total": {"value": 1},
            }
        }
        with pytest.raises(MlflowException, match="already exists"):
            store.create_experiment("dup")

    def test_create_experiment_empty_name_raises(self, store):
        with pytest.raises(MlflowException, match="Invalid experiment name"):
            store.create_experiment("")

    def test_get_experiment(self, store, mock_os_client):
        mock_os_client.get.return_value = {
            "_id": "1",
            "_source": {
                "experiment_id": "1",
                "name": "my-exp",
                "artifact_location": "/artifacts",
                "lifecycle_stage": "active",
                "creation_time": 1711089570000,
                "last_update_time": 1711089570000,
                "tags": [],
            },
        }
        exp = store.get_experiment("1")
        assert exp.name == "my-exp"
        assert exp.lifecycle_stage == "active"

    def test_delete_experiment(self, store, mock_os_client):
        mock_os_client.update.return_value = {"result": "updated"}
        store.delete_experiment("1")
        update_call = mock_os_client.update.call_args
        assert update_call[1]["body"]["doc"]["lifecycle_stage"] == "deleted"

    def test_restore_experiment(self, store, mock_os_client):
        mock_os_client.update.return_value = {"result": "updated"}
        store.restore_experiment("1")
        update_call = mock_os_client.update.call_args
        assert update_call[1]["body"]["doc"]["lifecycle_stage"] == "active"

    def test_rename_experiment(self, store, mock_os_client):
        # No duplicate name
        mock_os_client.search.return_value = {
            "hits": {"hits": [], "total": {"value": 0}},
        }
        mock_os_client.update.return_value = {"result": "updated"}
        store.rename_experiment("1", "new-name")
        update_call = mock_os_client.update.call_args
        assert update_call[1]["body"]["doc"]["name"] == "new-name"

    def test_search_experiments_active_only(self, store, mock_os_client):
        mock_os_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "experiment_id": "1",
                            "name": "exp1",
                            "lifecycle_stage": "active",
                            "creation_time": 1000,
                            "last_update_time": 1000,
                            "tags": [],
                        }
                    }
                ],
                "total": {"value": 1},
            }
        }
        results = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        assert len(results) == 1
        assert results[0].name == "exp1"


class TestRunCRUD:
    def test_create_run(self, store, mock_os_client):
        # Setup experiment get
        mock_os_client.get.return_value = {
            "_id": "1",
            "_source": {
                "experiment_id": "1",
                "name": "exp1",
                "artifact_location": "/artifacts/1",
                "lifecycle_stage": "active",
                "creation_time": 1000,
                "last_update_time": 1000,
                "tags": [],
            },
        }
        mock_os_client.index.return_value = {"_id": "run1", "result": "created"}
        mock_os_client.bulk.return_value = {"errors": False, "items": []}

        run = store.create_run("1", "user1", 1711089570000, [], "my-run")
        assert run.info.experiment_id == "1"
        assert run.info.user_id == "user1"
        assert run.info.run_name == "my-run"

    def test_get_run(self, store, mock_os_client):
        mock_os_client.get.return_value = {
            "_id": "run1",
            "_source": {
                "run_id": "run1",
                "experiment_id": "1",
                "user_id": "user1",
                "status": "FINISHED",
                "start_time": 1711089570000,
                "end_time": 1711089571000,
                "lifecycle_stage": "active",
                "artifact_uri": "/artifacts/1/run1",
                "run_name": "my-run",
            },
        }
        # Mock metrics, params, tags responses
        mock_os_client.search.return_value = {
            "hits": {"hits": [], "total": {"value": 0}},
            "aggregations": {"by_key": {"buckets": []}},
        }

        run = store.get_run("run1")
        assert run.info.run_id == "run1"
        assert run.info.status == "FINISHED"

    def test_delete_run(self, store, mock_os_client):
        mock_os_client.update.return_value = {"result": "updated"}
        store.delete_run("run1")
        update_call = mock_os_client.update.call_args
        assert update_call[1]["body"]["doc"]["lifecycle_stage"] == "deleted"

    def test_restore_run(self, store, mock_os_client):
        mock_os_client.update.return_value = {"result": "updated"}
        store.restore_run("run1")
        update_call = mock_os_client.update.call_args
        assert update_call[1]["body"]["doc"]["lifecycle_stage"] == "active"


class TestMetricsParamsTags:
    def test_log_metric(self, store, mock_os_client):
        mock_os_client.get.return_value = {
            "_id": "run1",
            "_source": {
                "run_id": "run1",
                "experiment_id": "1",
                "lifecycle_stage": "active",
                "status": "RUNNING",
                "start_time": 1711089570000,
                "user_id": "user1",
            },
        }
        mock_os_client.index.return_value = {"_id": "m1", "result": "created"}

        store.log_metric("run1", Metric("accuracy", 0.95, 1000, 0))
        assert mock_os_client.index.called

        # Verify the indexed doc has the right values
        call_args = mock_os_client.index.call_args
        assert call_args[1]["body"]["key"] == "accuracy"
        assert call_args[1]["body"]["value"] == 0.95

    def test_log_param(self, store, mock_os_client):
        mock_os_client.get.return_value = {
            "_id": "run1",
            "_source": {
                "run_id": "run1",
                "experiment_id": "1",
                "lifecycle_stage": "active",
                "status": "RUNNING",
                "start_time": 1711089570000,
                "user_id": "user1",
            },
        }
        mock_os_client.index.return_value = {"result": "created"}

        store.log_param("run1", Param("lr", "0.01"))
        call_args = mock_os_client.index.call_args
        assert call_args[1]["body"]["key"] == "lr"
        assert call_args[1]["body"]["value"] == "0.01"
        assert call_args[1]["id"] == "run1:lr"

    def test_set_tag(self, store, mock_os_client):
        mock_os_client.get.return_value = {
            "_id": "run1",
            "_source": {
                "run_id": "run1",
                "experiment_id": "1",
                "lifecycle_stage": "active",
                "status": "RUNNING",
                "start_time": 1711089570000,
                "user_id": "user1",
            },
        }
        mock_os_client.index.return_value = {"result": "created"}

        store.set_tag("run1", RunTag("env", "prod"))
        call_args = mock_os_client.index.call_args
        assert call_args[1]["body"]["key"] == "env"
        assert call_args[1]["body"]["value"] == "prod"

    def test_delete_tag(self, store, mock_os_client):
        mock_os_client.delete.return_value = {"result": "deleted"}
        store.delete_tag("run1", "env")
        mock_os_client.delete.assert_called_once()

    def test_log_batch(self, store, mock_os_client):
        mock_os_client.get.return_value = {
            "_id": "run1",
            "_source": {
                "run_id": "run1",
                "experiment_id": "1",
                "lifecycle_stage": "active",
                "status": "RUNNING",
                "start_time": 1711089570000,
                "user_id": "user1",
            },
        }
        mock_os_client.bulk.return_value = {"errors": False, "items": []}

        store.log_batch(
            "run1",
            metrics=[Metric("acc", 0.95, 1000, 0), Metric("loss", 0.05, 1000, 0)],
            params=[Param("lr", "0.01")],
            tags=[RunTag("env", "prod")],
        )
        assert mock_os_client.bulk.called


class TestTraceOperations:
    def test_start_trace(self, store, mock_os_client):
        mock_os_client.index.return_value = {"_id": "trace1", "result": "created"}

        class FakeTraceInfo:
            trace_id = "trace1"
            experiment_id = "1"
            request_time = 1711089570000
            state = "IN_PROGRESS"
            client_request_id = None
            request_preview = None
            response_preview = None

        store.start_trace(FakeTraceInfo())
        assert mock_os_client.index.called

    def test_set_trace_tag(self, store, mock_os_client):
        mock_os_client.index.return_value = {"result": "created"}
        store.set_trace_tag("trace1", "key1", "value1")
        call_args = mock_os_client.index.call_args
        assert call_args[1]["body"]["key"] == "key1"
        assert call_args[1]["body"]["value"] == "value1"

    def test_delete_trace_tag(self, store, mock_os_client):
        mock_os_client.delete.return_value = {"result": "deleted"}
        store.delete_trace_tag("trace1", "key1")
        mock_os_client.delete.assert_called_once()

    def test_delete_traces(self, store, mock_os_client):
        mock_os_client.delete_by_query.return_value = {"deleted": 5}
        store.delete_traces("1", 1711089570000)
        mock_os_client.delete_by_query.assert_called_once()

    def test_search_traces(self, store, mock_os_client):
        mock_os_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "trace_id": "t1",
                            "experiment_id": "1",
                            "status": "OK",
                            "request_time": 1711089570000,
                        }
                    }
                ],
                "total": {"value": 1},
            }
        }
        traces, token = store.search_traces(["1"])
        assert len(traces) == 1
        assert traces[0]["trace_id"] == "t1"


class TestExperimentTagOperations:
    def test_set_experiment_tag(self, store, mock_os_client):
        mock_os_client.update.return_value = {"result": "updated"}
        store.set_experiment_tag("1", ExperimentTag("key1", "value1"))
        mock_os_client.update.assert_called()
        call_args = mock_os_client.update.call_args
        assert "script" in call_args[1]["body"]
