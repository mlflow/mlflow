"""OpenSearch-backed tracking store for MLflow.

This module implements :class:`~mlflow.store.tracking.abstract_store.AbstractStore`
using OpenSearch as the persistence layer.  It is registered as a plugin via the
``mlflow.tracking_store`` entry-point group under the ``opensearch`` scheme.

Usage::

    mlflow server --backend-store-uri="opensearch://localhost:9200"

Or programmatically::

    import mlflow

    mlflow.set_tracking_uri("opensearch://localhost:9200")
"""

from __future__ import annotations

import logging
import math
import time
import uuid

from mlflow.entities import (
    Experiment,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.tracking._opensearch_client import OpenSearchClientManager
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.opensearch_query import (
    OpenSearchQueryTranslator,
    build_sort_clause,
)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.utils.uri import append_to_uri_path

_logger = logging.getLogger(__name__)

_DEFAULT_EXPERIMENT_ID = "0"


class OpenSearchTrackingStore(AbstractStore):
    """MLflow tracking store backed by OpenSearch.

    URI formats::

        opensearch://host:port[/index_prefix]
        opensearch+https://user:pass@host:port[/index_prefix]
    """

    SEARCH_MAX_RESULTS_DEFAULT = 1000

    def __init__(self, store_uri: str, artifact_uri: str | None = None):
        super().__init__()
        self._artifact_uri = artifact_uri
        self._manager = OpenSearchClientManager(store_uri)
        self._query_translator = OpenSearchQueryTranslator()
        self._ensure_default_experiment()

    def _ensure_default_experiment(self):
        """Create default experiment and required indices on first access."""
        self._manager.ensure_indices()

        try:
            self.get_experiment(_DEFAULT_EXPERIMENT_ID)
        except MlflowException:
            self._index_experiment(
                experiment_id=_DEFAULT_EXPERIMENT_ID,
                name=Experiment.DEFAULT_EXPERIMENT_NAME,
                artifact_location=self._artifact_uri or "",
                lifecycle_stage=LifecycleStage.ACTIVE,
                tags=[],
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _client(self):
        return self._manager.client

    def _idx(self, index_type: str) -> str:
        return self._manager.get_index_name(index_type)

    def _generate_experiment_id(self) -> str:
        """Generate a new unique experiment ID.

        Uses a counter document in OpenSearch for sequential IDs.
        """
        counter_index = self._idx("experiments")
        try:
            result = self._client.search(
                index=counter_index,
                body={
                    "size": 0,
                    "aggs": {"max_id": {"max": {"field": "experiment_id"}}},
                },
            )
            max_id = result["aggregations"]["max_id"]["value"]
            return str(int(max_id) + 1) if max_id is not None else "1"
        except Exception:
            return "1"

    def _index_experiment(
        self,
        experiment_id: str,
        name: str,
        artifact_location: str,
        lifecycle_stage: str,
        tags: list,
    ):
        now = int(time.time() * 1000)
        doc = {
            "experiment_id": experiment_id,
            "name": name,
            "artifact_location": artifact_location,
            "lifecycle_stage": lifecycle_stage,
            "creation_time": now,
            "last_update_time": now,
            "tags": [{"key": t.key, "value": t.value} for t in tags],
        }
        self._client.index(
            index=self._idx("experiments"),
            id=experiment_id,
            body=doc,
            refresh="wait_for",
        )

    def _doc_to_experiment(self, doc: dict) -> Experiment:
        src = doc if "_source" not in doc else doc["_source"]
        tags = [ExperimentTag(t["key"], t["value"]) for t in src.get("tags", [])]
        return Experiment(
            experiment_id=str(src["experiment_id"]),
            name=src["name"],
            artifact_location=src.get("artifact_location", ""),
            lifecycle_stage=src.get("lifecycle_stage", LifecycleStage.ACTIVE),
            tags=tags,
            creation_time=src.get("creation_time"),
            last_update_time=src.get("last_update_time"),
        )

    def _doc_to_run_info(self, src: dict) -> RunInfo:
        return RunInfo(
            run_id=src["run_id"],
            experiment_id=str(src["experiment_id"]),
            user_id=src.get("user_id", ""),
            status=src.get("status", RunStatus.to_string(RunStatus.RUNNING)),
            start_time=src.get("start_time"),
            end_time=src.get("end_time"),
            lifecycle_stage=src.get("lifecycle_stage", LifecycleStage.ACTIVE),
            artifact_uri=src.get("artifact_uri", ""),
            run_name=src.get("run_name", ""),
        )

    def _get_run_data(self, run_id: str) -> RunData:
        """Assemble RunData from metrics, params, and tags indices."""
        metrics = self._get_latest_metrics(run_id)
        params = self._get_params(run_id)
        tags = self._get_tags(run_id)
        return RunData(metrics=metrics, params=params, tags=tags)

    def _get_latest_metrics(self, run_id: str) -> list[Metric]:
        result = self._client.search(
            index=self._idx("metrics"),
            body={
                "query": {"term": {"run_id": run_id}},
                "size": 0,
                "aggs": {
                    "by_key": {
                        "terms": {"field": "key", "size": 10000},
                        "aggs": {
                            "latest": {
                                "top_hits": {
                                    "size": 1,
                                    "sort": [
                                        {"timestamp": {"order": "desc"}},
                                        {"step": {"order": "desc"}},
                                    ],
                                }
                            }
                        },
                    }
                },
            },
        )
        metrics = []
        for bucket in result["aggregations"]["by_key"]["buckets"]:
            hit = bucket["latest"]["hits"]["hits"][0]["_source"]
            val = hit["value"]
            if hit.get("is_nan"):
                val = float("nan")
            metrics.append(
                Metric(
                    key=hit["key"],
                    value=val,
                    timestamp=hit["timestamp"],
                    step=hit.get("step", 0),
                )
            )
        return metrics

    def _get_params(self, run_id: str) -> list[Param]:
        result = self._client.search(
            index=self._idx("params"),
            body={"query": {"term": {"run_id": run_id}}, "size": 10000},
        )
        return [
            Param(key=h["_source"]["key"], value=h["_source"]["value"])
            for h in result["hits"]["hits"]
        ]

    def _get_tags(self, run_id: str) -> list[RunTag]:
        result = self._client.search(
            index=self._idx("tags"),
            body={"query": {"term": {"run_id": run_id}}, "size": 10000},
        )
        return [
            RunTag(key=h["_source"]["key"], value=h["_source"]["value"])
            for h in result["hits"]["hits"]
        ]

    # ------------------------------------------------------------------
    # Experiment operations
    # ------------------------------------------------------------------

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        query_clauses = []

        # Lifecycle filter
        if view_type == ViewType.ACTIVE_ONLY:
            query_clauses.append({"term": {"lifecycle_stage": LifecycleStage.ACTIVE}})
        elif view_type == ViewType.DELETED_ONLY:
            query_clauses.append({"term": {"lifecycle_stage": LifecycleStage.DELETED}})

        # User filter
        if filter_string:
            translated = self._query_translator.translate(filter_string, entity_type="experiment")
            if "bool" in translated and "must" in translated["bool"]:
                query_clauses.extend(translated["bool"]["must"])

        body = {
            "query": {"bool": {"must": query_clauses}} if query_clauses else {"match_all": {}},
            "size": max_results,
        }

        sort = build_sort_clause(order_by)
        if sort:
            body["sort"] = sort

        if page_token:
            body["from"] = int(page_token)

        result = self._client.search(index=self._idx("experiments"), body=body)
        experiments = [self._doc_to_experiment(h) for h in result["hits"]["hits"]]

        next_token = None
        offset = int(page_token or 0)
        if offset + max_results < result["hits"]["total"]["value"]:
            next_token = str(offset + max_results)

        from mlflow.store.entities.paged_list import PagedList

        return PagedList(experiments, next_token)

    def create_experiment(self, name, artifact_location=None, tags=None):
        _validate_experiment_name(name)

        # Check for duplicate name
        existing = self.get_experiment_by_name(name)
        if existing is not None:
            raise MlflowException(
                f"Experiment(name={name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        experiment_id = self._generate_experiment_id()
        artifact_location = artifact_location or append_to_uri_path(
            self._artifact_uri or "", str(experiment_id)
        )

        self._index_experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location,
            lifecycle_stage=LifecycleStage.ACTIVE,
            tags=tags or [],
        )
        return experiment_id

    def get_experiment(self, experiment_id):
        try:
            result = self._client.get(index=self._idx("experiments"), id=str(experiment_id))
            return self._doc_to_experiment(result)
        except Exception as e:
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise MlflowException(
                    f"No Experiment with id={experiment_id} exists",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise

    def get_experiment_by_name(self, experiment_name):
        result = self._client.search(
            index=self._idx("experiments"),
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"name.keyword": experiment_name}},
                            {"term": {"lifecycle_stage": LifecycleStage.ACTIVE}},
                        ]
                    }
                },
                "size": 1,
            },
        )
        hits = result["hits"]["hits"]
        if not hits:
            return None
        return self._doc_to_experiment(hits[0])

    def delete_experiment(self, experiment_id):
        self.get_experiment(experiment_id)  # Verify exists
        self._client.update(
            index=self._idx("experiments"),
            id=str(experiment_id),
            body={
                "doc": {
                    "lifecycle_stage": LifecycleStage.DELETED,
                    "last_update_time": int(time.time() * 1000),
                }
            },
            refresh="wait_for",
        )

    def restore_experiment(self, experiment_id):
        self.get_experiment(experiment_id)  # Verify exists
        self._client.update(
            index=self._idx("experiments"),
            id=str(experiment_id),
            body={
                "doc": {
                    "lifecycle_stage": LifecycleStage.ACTIVE,
                    "last_update_time": int(time.time() * 1000),
                }
            },
            refresh="wait_for",
        )

    def rename_experiment(self, experiment_id, new_name):
        _validate_experiment_name(new_name)
        self.get_experiment(experiment_id)  # Verify exists

        existing = self.get_experiment_by_name(new_name)
        if existing is not None and existing.experiment_id != str(experiment_id):
            raise MlflowException(
                f"Experiment(name={new_name}) already exists.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )

        self._client.update(
            index=self._idx("experiments"),
            id=str(experiment_id),
            body={
                "doc": {
                    "name": new_name,
                    "last_update_time": int(time.time() * 1000),
                }
            },
            refresh="wait_for",
        )

    def set_experiment_tag(self, experiment_id, tag):
        self.get_experiment(experiment_id)  # Verify exists
        # Use painless script to upsert tag in nested array
        self._client.update(
            index=self._idx("experiments"),
            id=str(experiment_id),
            body={
                "script": {
                    "source": """
                        if (ctx._source.tags == null) {
                            ctx._source.tags = [];
                        }
                        def found = false;
                        for (int i = 0; i < ctx._source.tags.size(); i++) {
                            if (ctx._source.tags[i].key == params.key) {
                                ctx._source.tags[i].value = params.value;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            ctx._source.tags.add(params.tag);
                        }
                        ctx._source.last_update_time = params.now;
                    """,
                    "params": {
                        "key": tag.key,
                        "value": tag.value,
                        "tag": {"key": tag.key, "value": tag.value},
                        "now": int(time.time() * 1000),
                    },
                }
            },
            refresh="wait_for",
        )

    # ------------------------------------------------------------------
    # Run operations
    # ------------------------------------------------------------------

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                f"Experiment {experiment_id} is not active.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        run_id = uuid.uuid4().hex
        artifact_uri = append_to_uri_path(experiment.artifact_location, run_id, "artifacts")

        run_doc = {
            "run_id": run_id,
            "experiment_id": str(experiment_id),
            "user_id": user_id or "",
            "status": RunStatus.to_string(RunStatus.RUNNING),
            "start_time": start_time,
            "end_time": None,
            "lifecycle_stage": LifecycleStage.ACTIVE,
            "artifact_uri": artifact_uri,
            "run_name": run_name or "",
            "deleted_time": None,
        }

        self._client.index(
            index=self._idx("runs"),
            id=run_id,
            body=run_doc,
            refresh="wait_for",
        )

        # Log initial tags
        all_tags = list(tags or [])
        if run_name:
            all_tags.append(RunTag(MLFLOW_RUN_NAME, run_name))

        if all_tags:
            tag_docs = [{"run_id": run_id, "key": t.key, "value": t.value} for t in all_tags]
            self._manager.bulk_index(self._idx("tags"), tag_docs, id_field=None)

        run_info = self._doc_to_run_info(run_doc)
        run_data = RunData(tags=[RunTag(t.key, t.value) for t in all_tags])
        return Run(run_info, run_data)

    def update_run_info(self, run_id, run_status, end_time, run_name):
        doc_update = {}
        if run_status is not None:
            doc_update["status"] = RunStatus.to_string(run_status)
        if end_time is not None:
            doc_update["end_time"] = end_time
        if run_name is not None:
            doc_update["run_name"] = run_name

        self._client.update(
            index=self._idx("runs"),
            id=run_id,
            body={"doc": doc_update},
            refresh="wait_for",
        )
        return self.get_run(run_id).info

    def get_run(self, run_id):
        try:
            result = self._client.get(index=self._idx("runs"), id=run_id)
        except Exception as e:
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise MlflowException(
                    f"Run with id={run_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise

        run_info = self._doc_to_run_info(result["_source"])
        run_data = self._get_run_data(run_id)
        return Run(run_info, run_data)

    def delete_run(self, run_id):
        self._client.update(
            index=self._idx("runs"),
            id=run_id,
            body={
                "doc": {
                    "lifecycle_stage": LifecycleStage.DELETED,
                    "deleted_time": int(time.time() * 1000),
                }
            },
            refresh="wait_for",
        )

    def restore_run(self, run_id):
        self._client.update(
            index=self._idx("runs"),
            id=run_id,
            body={
                "doc": {
                    "lifecycle_stage": LifecycleStage.ACTIVE,
                    "deleted_time": None,
                }
            },
            refresh="wait_for",
        )

    def search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        query_clauses = [{"terms": {"experiment_id": [str(e) for e in experiment_ids]}}]

        # Lifecycle filter
        if run_view_type == ViewType.ACTIVE_ONLY:
            query_clauses.append({"term": {"lifecycle_stage": LifecycleStage.ACTIVE}})
        elif run_view_type == ViewType.DELETED_ONLY:
            query_clauses.append({"term": {"lifecycle_stage": LifecycleStage.DELETED}})

        # Collect run_id constraints from sub-queries (metrics, params, tags)
        run_id_sets = []
        if filter_string:
            translated = self._query_translator.translate(filter_string, entity_type="run")
            if "bool" in translated and "must" in translated["bool"]:
                for clause in translated["bool"]["must"]:
                    query_clauses.append(clause)

            # Handle sub-queries (cross-index)
            for sub in translated.get("_sub_queries", []):
                sub_index = self._idx(sub["_index"])
                return_field = sub["_return_field"]
                sub_result = self._client.search(
                    index=sub_index,
                    body={"query": sub["query"], "size": 10000, "_source": [return_field]},
                )
                ids = {h["_source"][return_field] for h in sub_result["hits"]["hits"]}
                run_id_sets.append(ids)

        # Intersect sub-query run_id sets
        if run_id_sets:
            valid_ids = set.intersection(*run_id_sets)
            if not valid_ids:
                from mlflow.store.entities.paged_list import PagedList

                return PagedList([], None)
            query_clauses.append({"terms": {"run_id": list(valid_ids)}})

        body = {
            "query": {"bool": {"must": query_clauses}},
            "size": max_results,
        }

        sort = build_sort_clause(order_by)
        if sort:
            body["sort"] = sort
        else:
            body["sort"] = [{"start_time": {"order": "desc"}}, {"run_id": {"order": "asc"}}]

        if page_token:
            body["from"] = int(page_token)

        result = self._client.search(index=self._idx("runs"), body=body)

        runs = []
        for hit in result["hits"]["hits"]:
            run_info = self._doc_to_run_info(hit["_source"])
            run_data = self._get_run_data(hit["_source"]["run_id"])
            runs.append(Run(run_info, run_data))

        next_token = None
        offset = int(page_token or 0)
        if offset + max_results < result["hits"]["total"]["value"]:
            next_token = str(offset + max_results)

        from mlflow.store.entities.paged_list import PagedList

        return PagedList(runs, next_token)

    # ------------------------------------------------------------------
    # Metrics / Params / Tags
    # ------------------------------------------------------------------

    def log_metric(self, run_id, metric):
        self.get_run(run_id)  # Verify run exists
        doc = {
            "run_id": run_id,
            "key": metric.key,
            "value": metric.value,
            "timestamp": metric.timestamp,
            "step": metric.step,
            "is_nan": math.isnan(metric.value) if isinstance(metric.value, float) else False,
        }
        self._client.index(
            index=self._idx("metrics"),
            body=doc,
            refresh="wait_for",
        )

    def log_batch(self, run_id, metrics=None, params=None, tags=None):
        self.get_run(run_id)  # Verify run exists

        if metrics:
            metric_docs = [
                {
                    "run_id": run_id,
                    "key": m.key,
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "step": m.step,
                    "is_nan": math.isnan(m.value) if isinstance(m.value, float) else False,
                }
                for m in metrics
            ]
            self._manager.bulk_index(self._idx("metrics"), metric_docs)

        if params:
            param_docs = [{"run_id": run_id, "key": p.key, "value": p.value} for p in params]
            self._manager.bulk_index(self._idx("params"), param_docs)

        if tags:
            tag_docs = [{"run_id": run_id, "key": t.key, "value": t.value} for t in tags]
            self._manager.bulk_index(self._idx("tags"), tag_docs)

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"run_id": run_id}},
                        {"term": {"key": metric_key}},
                    ]
                }
            },
            "sort": [
                {"timestamp": {"order": "asc"}},
                {"step": {"order": "asc"}},
            ],
            "size": max_results or 10000,
        }
        if page_token:
            body["from"] = int(page_token)

        result = self._client.search(index=self._idx("metrics"), body=body)

        from mlflow.store.entities.paged_list import PagedList

        metrics = []
        for h in result["hits"]["hits"]:
            src = h["_source"]
            val = src["value"]
            if src.get("is_nan"):
                val = float("nan")
            metrics.append(
                Metric(
                    key=src["key"],
                    value=val,
                    timestamp=src["timestamp"],
                    step=src.get("step", 0),
                )
            )

        next_token = None
        offset = int(page_token or 0)
        total = result["hits"]["total"]["value"]
        if offset + (max_results or 10000) < total:
            next_token = str(offset + (max_results or 10000))

        return PagedList(metrics, next_token)

    def log_param(self, run_id, param):
        self.get_run(run_id)  # Verify run exists
        doc_id = f"{run_id}:{param.key}"
        self._client.index(
            index=self._idx("params"),
            id=doc_id,
            body={"run_id": run_id, "key": param.key, "value": param.value},
            refresh="wait_for",
        )

    def set_tag(self, run_id, tag):
        self.get_run(run_id)  # Verify run exists
        doc_id = f"{run_id}:{tag.key}"
        self._client.index(
            index=self._idx("tags"),
            id=doc_id,
            body={"run_id": run_id, "key": tag.key, "value": tag.value},
            refresh="wait_for",
        )

    def delete_tag(self, run_id, key):
        doc_id = f"{run_id}:{key}"
        try:
            self._client.delete(
                index=self._idx("tags"),
                id=doc_id,
                refresh="wait_for",
            )
        except Exception as e:
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise MlflowException(
                    f"No tag with key={key} for run {run_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise

    # ------------------------------------------------------------------
    # Trace operations (skeleton — to be fully implemented in Phase 4)
    # ------------------------------------------------------------------

    def start_trace(self, trace_info):
        doc = {
            "trace_id": trace_info.trace_id,
            "experiment_id": str(trace_info.experiment_id),
            "request_time": trace_info.request_time,
            "status": str(trace_info.state),
            "execution_duration": None,
            "client_request_id": getattr(trace_info, "client_request_id", None),
            "request_preview": getattr(trace_info, "request_preview", None),
            "response_preview": getattr(trace_info, "response_preview", None),
        }
        self._client.index(
            index=self._idx("traces"),
            id=trace_info.trace_id,
            body=doc,
            refresh="wait_for",
        )
        return trace_info

    def end_trace(self, trace_id, trace_info):
        doc = {
            "status": str(trace_info.state),
            "execution_duration": getattr(trace_info, "execution_duration", None),
            "response_preview": getattr(trace_info, "response_preview", None),
        }
        self._client.update(
            index=self._idx("traces"),
            id=trace_id,
            body={"doc": doc},
            refresh="wait_for",
        )
        return trace_info

    def get_trace(self, trace_id, **kwargs):
        try:
            result = self._client.get(index=self._idx("traces"), id=trace_id)
            return result["_source"]
        except Exception as e:
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise MlflowException(
                    f"Trace with id={trace_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise

    def search_traces(
        self,
        experiment_ids,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        **kwargs,
    ):
        query_clauses = [{"terms": {"experiment_id": [str(e) for e in experiment_ids]}}]

        trace_id_sets = []
        if filter_string:
            translated = self._query_translator.translate(filter_string, entity_type="trace")
            if "bool" in translated and "must" in translated["bool"]:
                for clause in translated["bool"]["must"]:
                    query_clauses.append(clause)

            for sub in translated.get("_sub_queries", []):
                sub_index = self._idx(sub["_index"])
                return_field = sub["_return_field"]
                sub_result = self._client.search(
                    index=sub_index,
                    body={"query": sub["query"], "size": 10000, "_source": [return_field]},
                )
                ids = {h["_source"][return_field] for h in sub_result["hits"]["hits"]}
                trace_id_sets.append(ids)

        if trace_id_sets:
            valid_ids = set.intersection(*trace_id_sets)
            if not valid_ids:
                return [], None
            query_clauses.append({"terms": {"trace_id": list(valid_ids)}})

        body = {
            "query": {"bool": {"must": query_clauses}},
            "size": max_results,
            "sort": build_sort_clause(order_by) or [{"request_time": {"order": "desc"}}],
        }

        if page_token:
            body["from"] = int(page_token)

        result = self._client.search(index=self._idx("traces"), body=body)
        traces = [h["_source"] for h in result["hits"]["hits"]]

        next_token = None
        offset = int(page_token or 0)
        if offset + max_results < result["hits"]["total"]["value"]:
            next_token = str(offset + max_results)

        return traces, next_token

    def set_trace_tag(self, trace_id, key, value):
        doc_id = f"{trace_id}:{key}"
        self._client.index(
            index=self._idx("trace_tags"),
            id=doc_id,
            body={"trace_id": trace_id, "key": key, "value": value},
            refresh="wait_for",
        )

    def delete_trace_tag(self, trace_id, key):
        doc_id = f"{trace_id}:{key}"
        try:
            self._client.delete(
                index=self._idx("trace_tags"),
                id=doc_id,
                refresh="wait_for",
            )
        except Exception as e:
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise MlflowException(
                    f"No tag with key={key} for trace {trace_id}",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise

    def delete_traces(self, experiment_id, max_timestamp_millis, request_ids=None):
        query_clauses = [{"term": {"experiment_id": str(experiment_id)}}]
        if max_timestamp_millis:
            query_clauses.append({"range": {"request_time": {"lte": max_timestamp_millis}}})
        if request_ids:
            query_clauses.append({"terms": {"trace_id": request_ids}})

        self._client.delete_by_query(
            index=self._idx("traces"),
            body={"query": {"bool": {"must": query_clauses}}},
            refresh=True,
        )
        return 0

    # ------------------------------------------------------------------
    # Span operations
    # ------------------------------------------------------------------

    def log_spans(self, experiment_id, spans):
        span_docs = []
        for span in spans:
            doc = {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "experiment_id": str(experiment_id),
                "parent_span_id": getattr(span, "parent_span_id", None),
                "name": span.name,
                "type": getattr(span, "type", "UNKNOWN"),
                "status": str(getattr(span, "status", "")),
                "start_time_unix_nano": getattr(span, "start_time_unix_nano", None),
                "end_time_unix_nano": getattr(span, "end_time_unix_nano", None),
                "duration_ns": None,
                "content": str(getattr(span, "content", "")),
            }
            start = doc["start_time_unix_nano"]
            end = doc["end_time_unix_nano"]
            if start is not None and end is not None:
                doc["duration_ns"] = end - start
            span_docs.append(doc)

        self._manager.bulk_index(self._idx("spans"), span_docs)

    # ------------------------------------------------------------------
    # Assessment operations (skeleton)
    # ------------------------------------------------------------------

    def create_assessment(self, assessment):
        doc = {
            "assessment_id": getattr(assessment, "assessment_id", uuid.uuid4().hex),
            "trace_id": assessment.trace_id,
            "name": assessment.name,
            "create_time": int(time.time() * 1000),
            "last_update_time": int(time.time() * 1000),
        }
        self._client.index(
            index=self._idx("assessments"),
            id=doc["assessment_id"],
            body=doc,
            refresh="wait_for",
        )
        return assessment

    def get_assessment(self, trace_id, assessment_id):
        try:
            result = self._client.get(index=self._idx("assessments"), id=assessment_id)
            return result["_source"]
        except Exception as e:
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise MlflowException(
                    f"Assessment {assessment_id} not found",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                ) from e
            raise

    # ------------------------------------------------------------------
    # Stubs for remaining AbstractStore methods
    # ------------------------------------------------------------------

    def log_inputs(self, run_id, datasets=None):
        pass

    def record_logged_model(self, run_id, mlflow_model):
        pass


def _validate_experiment_name(name):
    if not name or not name.strip():
        raise MlflowException(
            "Invalid experiment name",
            error_code=INVALID_PARAMETER_VALUE,
        )


# Avoid circular import — ExperimentTag is light enough to import late.
try:
    from mlflow.entities import ExperimentTag
except ImportError:
    ExperimentTag = None
