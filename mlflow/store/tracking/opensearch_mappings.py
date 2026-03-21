"""OpenSearch index mapping definitions for MLflow tracking store.

This module defines the OpenSearch index mappings for all MLflow entities
(experiments, runs, metrics, params, tags, traces, spans, assessments).
Each entity maps to a dedicated index for better scaling and lifecycle management.
"""

# Default index prefix — can be overridden via URI path or env var.
DEFAULT_INDEX_PREFIX = "mlflow_"

# Default index settings
DEFAULT_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 1,
}

# High-volume index settings (metrics, spans)
HIGH_VOLUME_SETTINGS = {
    "number_of_shards": 3,
    "number_of_replicas": 1,
}

EXPERIMENTS_MAPPING = {
    "mappings": {
        "properties": {
            "experiment_id": {"type": "keyword"},
            "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "artifact_location": {"type": "keyword"},
            "lifecycle_stage": {"type": "keyword"},
            "creation_time": {"type": "long"},
            "last_update_time": {"type": "long"},
            "workspace": {"type": "keyword"},
            "tags": {
                "type": "nested",
                "properties": {
                    "key": {"type": "keyword"},
                    "value": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                },
            },
        }
    },
    "settings": DEFAULT_SETTINGS,
}

RUNS_MAPPING = {
    "mappings": {
        "properties": {
            "run_id": {"type": "keyword"},
            "run_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "experiment_id": {"type": "keyword"},
            "user_id": {"type": "keyword"},
            "status": {"type": "keyword"},
            "start_time": {"type": "long"},
            "end_time": {"type": "long"},
            "lifecycle_stage": {"type": "keyword"},
            "artifact_uri": {"type": "keyword"},
            "deleted_time": {"type": "long"},
        }
    },
    "settings": DEFAULT_SETTINGS,
}

METRICS_MAPPING = {
    "mappings": {
        "properties": {
            "run_id": {"type": "keyword"},
            "key": {"type": "keyword"},
            "value": {"type": "double"},
            "timestamp": {"type": "long"},
            "step": {"type": "long"},
            "is_nan": {"type": "boolean"},
            "model_id": {"type": "keyword"},
            "dataset_name": {"type": "keyword"},
            "dataset_digest": {"type": "keyword"},
        }
    },
    "settings": {
        **HIGH_VOLUME_SETTINGS,
        "index.sort.field": ["run_id", "key", "timestamp"],
        "index.sort.order": ["asc", "asc", "desc"],
    },
}

PARAMS_MAPPING = {
    "mappings": {
        "properties": {
            "run_id": {"type": "keyword"},
            "key": {"type": "keyword"},
            "value": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        }
    },
    "settings": DEFAULT_SETTINGS,
}

TAGS_MAPPING = {
    "mappings": {
        "properties": {
            "run_id": {"type": "keyword"},
            "key": {"type": "keyword"},
            "value": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        }
    },
    "settings": DEFAULT_SETTINGS,
}

TRACES_MAPPING = {
    "mappings": {
        "properties": {
            "trace_id": {"type": "keyword"},
            "experiment_id": {"type": "keyword"},
            "request_time": {"type": "long"},
            "execution_duration": {"type": "long"},
            "status": {"type": "keyword"},
            "client_request_id": {"type": "keyword"},
            "request_preview": {"type": "text"},
            "response_preview": {"type": "text"},
        }
    },
    "settings": DEFAULT_SETTINGS,
}

TRACE_TAGS_MAPPING = {
    "mappings": {
        "properties": {
            "trace_id": {"type": "keyword"},
            "key": {"type": "keyword"},
            "value": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        }
    },
    "settings": DEFAULT_SETTINGS,
}

TRACE_METADATA_MAPPING = {
    "mappings": {
        "properties": {
            "trace_id": {"type": "keyword"},
            "key": {"type": "keyword"},
            "value": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
        }
    },
    "settings": DEFAULT_SETTINGS,
}

SPANS_MAPPING = {
    "mappings": {
        "properties": {
            "trace_id": {"type": "keyword"},
            "span_id": {"type": "keyword"},
            "experiment_id": {"type": "keyword"},
            "parent_span_id": {"type": "keyword"},
            "name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "type": {"type": "keyword"},
            "status": {"type": "keyword"},
            "start_time_unix_nano": {"type": "long"},
            "end_time_unix_nano": {"type": "long"},
            "duration_ns": {"type": "long"},
            "content": {"type": "text"},
            "dimension_attributes": {"type": "object", "enabled": False},
        }
    },
    "settings": {
        **HIGH_VOLUME_SETTINGS,
        "analysis": {
            "analyzer": {
                "span_content_analyzer": {
                    "type": "standard",
                    "max_token_length": 255,
                }
            }
        },
    },
}

ASSESSMENTS_MAPPING = {
    "mappings": {
        "properties": {
            "assessment_id": {"type": "keyword"},
            "trace_id": {"type": "keyword"},
            "experiment_id": {"type": "keyword"},
            "name": {"type": "keyword"},
            "source": {"type": "object"},
            "create_time": {"type": "long"},
            "last_update_time": {"type": "long"},
            "evaluation": {
                "type": "object",
                "properties": {
                    "assessment_type": {"type": "keyword"},
                    "value": {"type": "text"},
                    "boolean_value": {"type": "boolean"},
                    "numeric_value": {"type": "double"},
                },
            },
            "rationale": {"type": "text"},
            "error": {"type": "object"},
            "metadata": {"type": "object", "enabled": False},
        }
    },
    "settings": DEFAULT_SETTINGS,
}


# Mapping of index suffix to its mapping definition
INDEX_MAPPINGS = {
    "experiments": EXPERIMENTS_MAPPING,
    "runs": RUNS_MAPPING,
    "metrics": METRICS_MAPPING,
    "params": PARAMS_MAPPING,
    "tags": TAGS_MAPPING,
    "traces": TRACES_MAPPING,
    "trace_tags": TRACE_TAGS_MAPPING,
    "trace_metadata": TRACE_METADATA_MAPPING,
    "spans": SPANS_MAPPING,
    "assessments": ASSESSMENTS_MAPPING,
}


def get_index_name(prefix: str, index_type: str) -> str:
    """Return the full index name for a given type.

    Args:
        prefix: Index prefix (e.g. "mlflow_").
        index_type: One of the keys in INDEX_MAPPINGS.

    Returns:
        Full index name, e.g. "mlflow_experiments".
    """
    return f"{prefix}{index_type}"


def get_all_index_configs(prefix: str) -> dict[str, dict]:
    """Return a mapping of full index names to their configurations.

    Args:
        prefix: Index prefix (e.g. "mlflow_").

    Returns:
        Dict mapping index name → mapping body.
    """
    return {
        get_index_name(prefix, idx_type): mapping for idx_type, mapping in INDEX_MAPPINGS.items()
    }
