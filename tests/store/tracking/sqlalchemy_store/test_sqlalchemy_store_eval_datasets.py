import json
import time
from unittest import mock

import pytest

from mlflow.entities.dataset_record import DatasetRecord
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlEvaluationDatasetRecord
from mlflow.utils import mlflow_tags

from tests.store.tracking.sqlalchemy_store.conftest import (
    _create_experiments,
)

pytestmark = pytest.mark.notrackingurimock


def test_dataset_crud_operations(store):
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        experiment_ids = _create_experiments(store, ["test_exp_1", "test_exp_2"])
        created_dataset = store.create_dataset(
            name="test_eval_dataset",
            tags={
                "purpose": "testing",
                "environment": "test",
                mlflow_tags.MLFLOW_USER: "test_user",
            },
            experiment_ids=experiment_ids,
        )

        assert created_dataset.dataset_id is not None
        assert created_dataset.dataset_id.startswith("d-")
        assert created_dataset.name == "test_eval_dataset"
        assert created_dataset.tags == {
            "purpose": "testing",
            "environment": "test",
            mlflow_tags.MLFLOW_USER: "test_user",
        }
        assert created_dataset.created_time > 0
        assert created_dataset.last_update_time > 0
        assert created_dataset.created_time == created_dataset.last_update_time
        assert created_dataset.schema is None  # Schema is computed when data is added
        assert created_dataset.profile is None  # Profile is computed when data is added
        assert created_dataset.created_by == "test_user"  # Extracted from mlflow.user tag

        retrieved_dataset = store.get_dataset(dataset_id=created_dataset.dataset_id)
        assert retrieved_dataset.dataset_id == created_dataset.dataset_id
        assert retrieved_dataset.name == created_dataset.name
        assert retrieved_dataset.tags == created_dataset.tags
        assert retrieved_dataset._experiment_ids is None
        assert retrieved_dataset.experiment_ids == experiment_ids
        assert not retrieved_dataset.has_records()

        with pytest.raises(
            MlflowException, match="Evaluation dataset with id 'd-nonexistent' not found"
        ):
            store.get_dataset(dataset_id="d-nonexistent")

        store.delete_dataset(created_dataset.dataset_id)
        with pytest.raises(MlflowException, match="not found"):
            store.get_dataset(dataset_id=created_dataset.dataset_id)

        # Verify idempotency
        store.delete_dataset("d-nonexistent")


def test_dataset_records_pagination(store):
    exp_id = _create_experiments(store, ["pagination_test_exp"])[0]

    dataset = store.create_dataset(
        name="pagination_test_dataset", experiment_ids=[exp_id], tags={"test": "pagination"}
    )

    records = [
        {
            "inputs": {"id": i, "question": f"Question {i}"},
            "expectations": {"answer": f"Answer {i}"},
            "tags": {"index": str(i)},
        }
        for i in range(25)
    ]

    store.upsert_dataset_records(dataset.dataset_id, records)

    page1, next_token1 = store._load_dataset_records(dataset.dataset_id, max_results=10)
    assert len(page1) == 10
    assert next_token1 is not None  # Token should exist for more pages

    # Collect all IDs from page1
    page1_ids = {r.inputs["id"] for r in page1}
    assert len(page1_ids) == 10  # All IDs should be unique

    page2, next_token2 = store._load_dataset_records(
        dataset.dataset_id, max_results=10, page_token=next_token1
    )
    assert len(page2) == 10
    assert next_token2 is not None  # Token should exist for more pages

    # Collect all IDs from page2
    page2_ids = {r.inputs["id"] for r in page2}
    assert len(page2_ids) == 10  # All IDs should be unique
    assert page1_ids.isdisjoint(page2_ids)  # No overlap between pages

    page3, next_token3 = store._load_dataset_records(
        dataset.dataset_id, max_results=10, page_token=next_token2
    )
    assert len(page3) == 5
    assert next_token3 is None  # No more pages

    # Collect all IDs from page3
    page3_ids = {r.inputs["id"] for r in page3}
    assert len(page3_ids) == 5  # All IDs should be unique
    assert page1_ids.isdisjoint(page3_ids)  # No overlap
    assert page2_ids.isdisjoint(page3_ids)  # No overlap

    # Verify we got all 25 records across all pages
    all_ids = page1_ids | page2_ids | page3_ids
    assert all_ids == set(range(25))

    all_records, no_token = store._load_dataset_records(dataset.dataset_id, max_results=None)
    assert len(all_records) == 25
    assert no_token is None

    # Verify we have all expected records (order doesn't matter)
    all_record_ids = {r.inputs["id"] for r in all_records}
    assert all_record_ids == set(range(25))


def test_dataset_search_comprehensive(store):
    test_prefix = "test_search_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp_{i}" for i in range(1, 4)])

    datasets = []
    for i in range(10):
        name = f"{test_prefix}dataset_{i:02d}"
        tags = {
            "priority": "high" if i % 2 == 0 else "low",
            mlflow_tags.MLFLOW_USER: f"user_{i % 3}",
        }

        if i < 3:
            created = store.create_dataset(
                name=name,
                experiment_ids=[exp_ids[0]],
                tags=tags,
            )
        elif i < 6:
            created = store.create_dataset(
                name=name,
                experiment_ids=[exp_ids[1], exp_ids[2]],
                tags=tags,
            )
        elif i < 8:
            created = store.create_dataset(
                name=name,
                experiment_ids=[exp_ids[2]],
                tags=tags,
            )
        else:
            created = store.create_dataset(
                name=name,
                experiment_ids=[],
                tags=tags,
            )
        datasets.append(created)
        time.sleep(0.001)

    results = store.search_datasets(experiment_ids=[exp_ids[0]])
    assert len([d for d in results if d.name.startswith(test_prefix)]) == 3

    results = store.search_datasets(experiment_ids=[exp_ids[1], exp_ids[2]])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 5

    results = store.search_datasets(order_by=["name"])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    names = [d.name for d in test_results]
    assert names == sorted(names)

    results = store.search_datasets(order_by=["name DESC"])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    names = [d.name for d in test_results]
    assert names == sorted(names, reverse=True)

    page1 = store.search_datasets(max_results=3)
    assert len(page1) == 3
    assert page1.token is not None

    page2 = store.search_datasets(max_results=3, page_token=page1.token)
    assert len(page2) == 3
    assert all(d1.dataset_id != d2.dataset_id for d1 in page1 for d2 in page2)

    results = store.search_datasets(experiment_ids=None)
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 10

    results = store.search_datasets(filter_string=f"name LIKE '%{test_prefix}dataset_0%'")
    assert len(results) == 10
    assert all("dataset_0" in d.name for d in results)

    results = store.search_datasets(filter_string=f"name = '{test_prefix}dataset_05'")
    assert len(results) == 1
    assert results[0].name == f"{test_prefix}dataset_05"

    results = store.search_datasets(filter_string="tags.priority = 'high'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 5
    assert all(d.tags.get("priority") == "high" for d in test_results)

    results = store.search_datasets(filter_string="tags.priority != 'high'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 5
    assert all(d.tags.get("priority") == "low" for d in test_results)

    results = store.search_datasets(
        filter_string=f"name LIKE '%{test_prefix}%' AND tags.priority = 'low'"
    )
    assert len(results) == 5
    assert all(d.tags.get("priority") == "low" and test_prefix in d.name for d in results)

    mid_dataset = datasets[5]
    results = store.search_datasets(filter_string=f"created_time > {mid_dataset.created_time}")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 4
    assert all(d.created_time > mid_dataset.created_time for d in test_results)

    results = store.search_datasets(
        experiment_ids=[exp_ids[0]], filter_string="tags.priority = 'high'"
    )
    assert len(results) == 2
    assert all(d.tags.get("priority") == "high" for d in results)

    results = store.search_datasets(filter_string="tags.priority = 'low'", order_by=["name ASC"])
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    names = [d.name for d in test_results]
    assert names == sorted(names)

    created_user = store.create_dataset(
        name=f"{test_prefix}_user_dataset",
        tags={"test": "user", mlflow_tags.MLFLOW_USER: "test_user_1"},
        experiment_ids=[exp_ids[0]],
    )

    results = store.search_datasets(filter_string="created_by = 'test_user_1'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 1
    assert test_results[0].created_by == "test_user_1"

    records_with_user = [
        {
            "inputs": {"test": "data"},
            "expectations": {"result": "expected"},
            "tags": {mlflow_tags.MLFLOW_USER: "test_user_2"},
        }
    ]
    store.upsert_dataset_records(created_user.dataset_id, records_with_user)

    results = store.search_datasets(filter_string="last_updated_by = 'test_user_2'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 1
    assert test_results[0].last_updated_by == "test_user_2"

    with pytest.raises(MlflowException, match="Invalid attribute key"):
        store.search_datasets(filter_string="invalid_field = 'value'")


def test_dataset_schema_and_profile_computation(store):
    test_prefix = "test_schema_profile_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    assert dataset.schema is None
    assert dataset.profile is None

    records = [
        {
            "inputs": {
                "question": "What is MLflow?",
                "temperature": 0.7,
                "max_tokens": 100,
                "use_cache": True,
                "tags": ["ml", "tools"],
            },
            "expectations": {
                "accuracy": 0.95,
                "contains_key_info": True,
                "response": "MLflow is an open source platform",
            },
            "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace1"}},
        },
        {
            "inputs": {
                "question": "What is Python?",
                "temperature": 0.5,
                "max_tokens": 150,
                "metadata": {"user": "test", "session": 123},
            },
            "expectations": {"accuracy": 0.9},
            "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace2"}},
        },
        {
            "inputs": {"question": "What is Docker?", "temperature": 0.8},
            "source": {"source_type": "HUMAN", "source_data": {"user": "human"}},
        },
    ]

    store.upsert_dataset_records(dataset.dataset_id, records)

    updated_dataset = store.get_dataset(dataset.dataset_id)

    assert updated_dataset.schema is not None
    schema = json.loads(updated_dataset.schema)
    assert "inputs" in schema
    assert "expectations" in schema
    assert schema["inputs"]["question"] == "string"
    assert schema["inputs"]["temperature"] == "float"
    assert schema["inputs"]["max_tokens"] == "integer"
    assert schema["inputs"]["use_cache"] == "boolean"
    assert schema["inputs"]["tags"] == "array"
    assert schema["inputs"]["metadata"] == "object"
    assert schema["expectations"]["accuracy"] == "float"
    assert schema["expectations"]["contains_key_info"] == "boolean"
    assert schema["expectations"]["response"] == "string"

    assert updated_dataset.profile is not None
    profile = json.loads(updated_dataset.profile)
    assert profile["num_records"] == 3


def test_dataset_schema_and_profile_incremental_updates(store):
    test_prefix = "test_incremental_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    initial_records = [
        {
            "inputs": {"question": "What is MLflow?", "temperature": 0.7},
            "expectations": {"accuracy": 0.95},
            "source": {"source_type": "TRACE", "source_data": {"trace_id": "trace1"}},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, initial_records)

    dataset1 = store.get_dataset(dataset.dataset_id)
    schema1 = json.loads(dataset1.schema)
    profile1 = json.loads(dataset1.profile)

    assert schema1["inputs"] == {"question": "string", "temperature": "float"}
    assert schema1["expectations"] == {"accuracy": "float"}
    assert profile1["num_records"] == 1

    additional_records = [
        {
            "inputs": {
                "question": "What is Python?",
                "temperature": 0.5,
                "max_tokens": 100,
                "use_cache": True,
            },
            "expectations": {"accuracy": 0.9, "relevance": 0.85},
            "source": {"source_type": "HUMAN", "source_data": {"user": "test_user"}},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, additional_records)

    dataset2 = store.get_dataset(dataset.dataset_id)
    schema2 = json.loads(dataset2.schema)
    profile2 = json.loads(dataset2.profile)

    assert schema2["inputs"]["question"] == "string"
    assert schema2["inputs"]["temperature"] == "float"
    assert schema2["inputs"]["max_tokens"] == "integer"
    assert schema2["inputs"]["use_cache"] == "boolean"
    assert schema2["expectations"]["accuracy"] == "float"
    assert schema2["expectations"]["relevance"] == "float"

    assert profile2["num_records"] == 2


def test_dataset_user_detection(store):
    test_prefix = "test_user_detection_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset1 = store.create_dataset(
        name=f"{test_prefix}dataset1",
        tags={mlflow_tags.MLFLOW_USER: "john_doe", "other": "tag"},
        experiment_ids=exp_ids,
    )
    assert dataset1.created_by == "john_doe"
    assert dataset1.tags[mlflow_tags.MLFLOW_USER] == "john_doe"

    dataset2 = store.create_dataset(
        name=f"{test_prefix}dataset2", tags={"other": "tag"}, experiment_ids=exp_ids
    )
    assert dataset2.created_by is None
    assert mlflow_tags.MLFLOW_USER not in dataset2.tags

    results = store.search_datasets(filter_string="created_by = 'john_doe'")
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 1
    assert test_results[0].dataset_id == dataset1.dataset_id


def test_dataset_filtering_ordering_pagination(store):
    test_prefix = "test_filter_order_page_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp_{i}" for i in range(3)])

    datasets = []
    for i in range(10):
        time.sleep(0.01)
        tags = {
            "priority": "high" if i < 3 else ("medium" if i < 7 else "low"),
            "model": f"model_{i % 3}",
            "environment": "production" if i % 2 == 0 else "staging",
        }
        created = store.create_dataset(
            name=f"{test_prefix}_dataset_{i:02d}",
            tags=tags,
            experiment_ids=[exp_ids[i % len(exp_ids)]],
        )
        datasets.append(created)

    results = store.search_datasets(
        filter_string="tags.priority = 'high'", order_by=["name ASC"], max_results=2
    )
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert len(test_results) == 2
    assert all(d.tags.get("priority") == "high" for d in test_results)
    assert test_results[0].name < test_results[1].name

    results_all = store.search_datasets(
        filter_string="tags.priority = 'high'", order_by=["name ASC"]
    )
    test_results_all = [d for d in results_all if d.name.startswith(test_prefix)]
    assert len(test_results_all) == 3

    mid_time = datasets[5].created_time
    results = store.search_datasets(
        filter_string=f"tags.environment = 'production' AND created_time > {mid_time}",
        order_by=["created_time DESC"],
        max_results=3,
    )
    test_results = [d for d in results if d.name.startswith(test_prefix)]
    assert all(d.tags.get("environment") == "production" for d in test_results)
    assert all(d.created_time > mid_time for d in test_results)

    for i in range(1, len(test_results)):
        assert test_results[i - 1].created_time >= test_results[i].created_time

    results = store.search_datasets(
        experiment_ids=[exp_ids[0]],
        filter_string="tags.model = 'model_0' AND tags.priority != 'low'",
        order_by=["last_update_time DESC"],
        max_results=5,
    )
    for d in results:
        assert d.tags.get("model") == "model_0"
        assert d.tags.get("priority") != "low"

    all_production = store.search_datasets(
        filter_string="tags.environment = 'production'", order_by=["name ASC"]
    )
    test_all_production = [d for d in all_production if d.name.startswith(test_prefix)]

    limited_results = store.search_datasets(
        filter_string="tags.environment = 'production'", order_by=["name ASC"], max_results=3
    )
    test_limited = [d for d in limited_results if d.name.startswith(test_prefix)]

    assert len(test_limited) == 3
    assert len(test_all_production) == 5
    for i in range(3):
        assert test_limited[i].dataset_id == test_all_production[i].dataset_id


def test_dataset_upsert_comprehensive(store):
    created_dataset = store.create_dataset(name="upsert_comprehensive")

    records_batch1 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is a platform", "score": 0.8},
            "tags": {"version": "v1", "quality": "high"},
            "source": {
                "source_type": "TRACE",
                "source_data": {"trace_id": "trace-001", "span_id": "span-001"},
            },
        },
        {
            "inputs": {"question": "What is Python?"},
            "expectations": {"answer": "Python is a language"},
            "tags": {"category": "programming"},
        },
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is an ML platform", "confidence": 0.9},
            "tags": {"version": "v2", "reviewed": "true"},
            "source": {
                "source_type": "TRACE",
                "source_data": {"trace_id": "trace-002", "span_id": "span-002"},
            },
        },
    ]

    result = store.upsert_dataset_records(created_dataset.dataset_id, records_batch1)
    assert result["inserted"] == 2
    assert result["updated"] == 1

    loaded_records, next_token = store._load_dataset_records(created_dataset.dataset_id)
    assert len(loaded_records) == 2
    assert next_token is None

    mlflow_record = next(r for r in loaded_records if r.inputs["question"] == "What is MLflow?")
    assert mlflow_record.expectations == {
        "answer": "MLflow is an ML platform",
        "score": 0.8,
        "confidence": 0.9,
    }
    assert mlflow_record.tags == {"version": "v2", "quality": "high", "reviewed": "true"}

    assert mlflow_record.source.source_type == "TRACE"
    assert mlflow_record.source.source_data["trace_id"] == "trace-001"
    assert mlflow_record.source_id == "trace-001"

    initial_update_time = mlflow_record.last_update_time
    time.sleep(0.01)

    records_batch2 = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"answer": "MLflow is the best ML platform", "rating": 5},
            "tags": {"version": "v3"},
        },
        {
            "inputs": {"question": "What is Spark?"},
            "expectations": {"answer": "Spark is a data processing engine"},
        },
    ]

    result = store.upsert_dataset_records(created_dataset.dataset_id, records_batch2)
    assert result["inserted"] == 1
    assert result["updated"] == 1

    loaded_records, next_token = store._load_dataset_records(created_dataset.dataset_id)
    assert len(loaded_records) == 3
    assert next_token is None

    updated_mlflow_record = next(
        r for r in loaded_records if r.inputs["question"] == "What is MLflow?"
    )
    assert updated_mlflow_record.expectations == {
        "answer": "MLflow is the best ML platform",
        "score": 0.8,
        "confidence": 0.9,
        "rating": 5,
    }
    assert updated_mlflow_record.tags == {
        "version": "v3",
        "quality": "high",
        "reviewed": "true",
    }
    assert updated_mlflow_record.last_update_time > initial_update_time
    assert updated_mlflow_record.source.source_data["trace_id"] == "trace-001"

    records_batch3 = [
        {"inputs": {"minimal": "input"}, "expectations": {"result": "minimal test"}},
        {"inputs": {"question": "Empty expectations"}, "expectations": {}},
        {"inputs": {"question": "No tags"}, "expectations": {"answer": "No tags"}, "tags": {}},
    ]

    result = store.upsert_dataset_records(created_dataset.dataset_id, records_batch3)
    assert result["inserted"] == 3
    assert result["updated"] == 0

    result = store.upsert_dataset_records(
        created_dataset.dataset_id,
        [{"inputs": {}, "expectations": {"result": "empty inputs allowed"}}],
    )
    assert result["inserted"] == 1
    assert result["updated"] == 0

    empty_result = store.upsert_dataset_records(created_dataset.dataset_id, [])
    assert empty_result["inserted"] == 0
    assert empty_result["updated"] == 0


def test_dataset_delete_records(store):
    test_prefix = "test_delete_records_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    records = [
        {
            "inputs": {"id": 1, "question": "What is MLflow?"},
            "expectations": {"answer": "ML platform"},
        },
        {
            "inputs": {"id": 2, "question": "What is Python?"},
            "expectations": {"answer": "Programming language"},
        },
        {
            "inputs": {"id": 3, "question": "What is Docker?"},
            "expectations": {"answer": "Container platform"},
        },
    ]
    store.upsert_dataset_records(dataset.dataset_id, records)

    loaded_records, _ = store._load_dataset_records(dataset.dataset_id)
    assert len(loaded_records) == 3

    record_ids = [r.dataset_record_id for r in loaded_records]

    deleted_count = store.delete_dataset_records(dataset.dataset_id, [record_ids[0]])
    assert deleted_count == 1

    remaining_records, _ = store._load_dataset_records(dataset.dataset_id)
    assert len(remaining_records) == 2

    updated_dataset = store.get_dataset(dataset.dataset_id)
    profile = json.loads(updated_dataset.profile)
    assert profile["num_records"] == 2

    deleted_count = store.delete_dataset_records(dataset.dataset_id, [record_ids[1], record_ids[2]])
    assert deleted_count == 2

    final_records, _ = store._load_dataset_records(dataset.dataset_id)
    assert len(final_records) == 0


def test_dataset_delete_records_idempotent(store):
    test_prefix = "test_delete_idempotent_"
    exp_ids = _create_experiments(store, [f"{test_prefix}exp"])

    dataset = store.create_dataset(name=f"{test_prefix}dataset", experiment_ids=exp_ids)

    deleted_count = store.delete_dataset_records(dataset.dataset_id, ["nonexistent-record-id"])
    assert deleted_count == 0


def test_dataset_associations_and_lazy_loading(store):
    experiment_ids = _create_experiments(store, ["test_exp_1", "test_exp_2", "test_exp_3"])
    created_dataset = store.create_dataset(
        name="multi_exp_dataset",
        experiment_ids=experiment_ids,
    )

    retrieved = store.get_dataset(dataset_id=created_dataset.dataset_id)
    assert retrieved._experiment_ids is None
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert set(retrieved.experiment_ids) == set(experiment_ids)

    results = store.search_datasets(experiment_ids=[experiment_ids[1]])
    assert any(d.dataset_id == created_dataset.dataset_id for d in results)

    results = store.search_datasets(experiment_ids=[experiment_ids[0], experiment_ids[2]])
    matching = [d for d in results if d.dataset_id == created_dataset.dataset_id]
    assert len(matching) == 1
    assert matching[0]._experiment_ids is None
    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        assert set(matching[0].experiment_ids) == set(experiment_ids)

    records = [{"inputs": {"q": f"Q{i}"}, "expectations": {"a": f"A{i}"}} for i in range(5)]
    store.upsert_dataset_records(created_dataset.dataset_id, records)

    with mock.patch("mlflow.tracking._tracking_service.utils._get_store", return_value=store):
        retrieved = store.get_dataset(dataset_id=created_dataset.dataset_id)
        assert not retrieved.has_records()

        df = retrieved.to_df()
        assert len(df) == 5
        assert retrieved.has_records()

        assert list(df.columns) == [
            "inputs",
            "outputs",
            "expectations",
            "tags",
            "source_type",
            "source_id",
            "source",
            "created_time",
            "dataset_record_id",
        ]


def test_dataset_get_experiment_ids(store):
    experiment_ids = _create_experiments(store, ["exp_1", "exp_2", "exp_3"])
    created_dataset = store.create_dataset(
        name="test_get_experiment_ids",
        experiment_ids=experiment_ids,
    )

    fetched_experiment_ids = store.get_dataset_experiment_ids(created_dataset.dataset_id)
    assert set(fetched_experiment_ids) == set(experiment_ids)

    created_dataset2 = store.create_dataset(
        name="test_no_experiments",
        experiment_ids=[],
    )
    fetched_experiment_ids2 = store.get_dataset_experiment_ids(created_dataset2.dataset_id)
    assert fetched_experiment_ids2 == []

    result = store.get_dataset_experiment_ids("d-nonexistent")
    assert result == []

    result = store.get_dataset_experiment_ids("")
    assert result == []


def test_dataset_tags_with_sql_backend(store):
    tags = {"environment": "production", "version": "2.0", "team": "ml-ops"}

    created = store.create_dataset(
        name="tagged_dataset",
        tags=tags,
    )
    assert created.tags == tags

    retrieved = store.get_dataset(created.dataset_id)
    assert retrieved.tags == tags
    assert retrieved.tags["environment"] == "production"
    assert retrieved.tags["version"] == "2.0"
    assert retrieved.tags["team"] == "ml-ops"

    created_none = store.create_dataset(
        name="no_tags_dataset",
        tags=None,
    )
    retrieved_none = store.get_dataset(created_none.dataset_id)
    assert retrieved_none.tags == {}

    created_empty = store.create_dataset(
        name="empty_tags_dataset",
        tags={},
        experiment_ids=None,
    )
    retrieved_empty = store.get_dataset(created_empty.dataset_id)
    assert retrieved_empty.tags == {}


def test_dataset_update_tags(store):
    initial_tags = {"environment": "development", "version": "1.0", "deprecated": "true"}
    created = store.create_dataset(
        name="test_update_tags",
        tags=initial_tags,
        experiment_ids=None,
    )

    retrieved = store.get_dataset(created.dataset_id)
    assert retrieved.tags == initial_tags

    update_tags = {
        "environment": "production",
        "team": "ml-ops",
        "deprecated": None,  # This will be ignored, not delete the tag
    }
    store.set_dataset_tags(created.dataset_id, update_tags)

    updated = store.get_dataset(created.dataset_id)
    expected_tags = {
        "environment": "production",  # Updated
        "version": "1.0",  # Preserved
        "deprecated": "true",  # Preserved (None didn't delete it)
        "team": "ml-ops",  # Added
    }
    assert updated.tags == expected_tags
    assert updated.last_update_time == created.last_update_time
    assert updated.last_updated_by == created.last_updated_by

    created_no_tags = store.create_dataset(
        name="test_no_initial_tags",
        tags=None,
        experiment_ids=None,
    )

    store.set_dataset_tags(
        created_no_tags.dataset_id,
        {"new_tag": "value", mlflow_tags.MLFLOW_USER: "test_user2"},
    )

    updated_no_tags = store.get_dataset(created_no_tags.dataset_id)
    assert updated_no_tags.tags == {"new_tag": "value", mlflow_tags.MLFLOW_USER: "test_user2"}
    assert updated_no_tags.last_update_time == created_no_tags.last_update_time
    assert updated_no_tags.last_updated_by == created_no_tags.last_updated_by


def test_dataset_digest_updates_with_changes(store):
    experiment_id = store.create_experiment("test_exp")

    dataset = store.create_dataset(
        name="test_dataset",
        tags={"env": "test"},
        experiment_ids=[experiment_id],
    )

    initial_digest = dataset.digest
    assert initial_digest is not None

    time.sleep(0.01)  # Ensure time difference

    records = [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {"accuracy": 0.95},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, records)

    updated_dataset = store.get_dataset(dataset.dataset_id)

    assert updated_dataset.digest != initial_digest

    prev_digest = updated_dataset.digest
    time.sleep(0.01)  # Ensure time difference

    more_records = [
        {
            "inputs": {"question": "How to track experiments?"},
            "expectations": {"accuracy": 0.9},
        }
    ]

    store.upsert_dataset_records(dataset.dataset_id, more_records)

    final_dataset = store.get_dataset(dataset.dataset_id)

    assert final_dataset.digest != prev_digest
    assert final_dataset.digest != initial_digest

    store.set_dataset_tags(dataset.dataset_id, {"new_tag": "value"})
    dataset_after_tags = store.get_dataset(dataset.dataset_id)

    assert dataset_after_tags.digest == final_dataset.digest


def test_sql_dataset_record_merge():
    with mock.patch("mlflow.store.tracking.dbmodels.models.get_current_time_millis") as mock_time:
        mock_time.return_value = 2000

        record = SqlEvaluationDatasetRecord()
        record.expectations = {"accuracy": 0.8, "relevance": 0.7}
        record.tags = {"env": "test"}
        record.created_time = 1000
        record.last_update_time = 1000
        record.created_by = "user1"
        record.last_updated_by = "user1"

        new_data = {
            "expectations": {"accuracy": 0.9, "completeness": 0.95},
            "tags": {"version": "2.0"},
        }

        record.merge(new_data)

        assert record.expectations == {
            "accuracy": 0.9,  # Updated
            "relevance": 0.7,  # Preserved
            "completeness": 0.95,  # Added
        }

        assert record.tags == {
            "env": "test",  # Preserved
            "version": "2.0",  # Added
        }

        assert record.created_time == 1000  # Preserved
        assert record.last_update_time == 2000  # Updated

        assert record.created_by == "user1"  # Preserved
        assert record.last_updated_by == "user1"  # No mlflow.user in tags

        record2 = SqlEvaluationDatasetRecord()
        record2.expectations = None
        record2.tags = None

        new_data2 = {"expectations": {"accuracy": 0.9}, "tags": {"env": "prod"}}

        record2.merge(new_data2)

        assert record2.expectations == {"accuracy": 0.9}
        assert record2.tags == {"env": "prod"}
        assert record2.last_update_time == 2000

        record3 = SqlEvaluationDatasetRecord()
        record3.created_by = "user1"
        record3.last_updated_by = "user1"

        new_data3 = {"tags": {mlflow_tags.MLFLOW_USER: "user2", "env": "prod"}}

        record3.merge(new_data3)

        assert record3.created_by == "user1"  # Preserved
        assert record3.last_updated_by == "user2"  # Updated from mlflow.user tag

        record4 = SqlEvaluationDatasetRecord()
        record4.expectations = {"accuracy": 0.8}
        record4.tags = {"env": "test"}
        record4.last_update_time = 1000

        record4.merge({})

        assert record4.expectations == {"accuracy": 0.8}
        assert record4.tags == {"env": "test"}
        assert record4.last_update_time == 2000

        record5 = SqlEvaluationDatasetRecord()
        record5.expectations = {"accuracy": 0.8}
        record5.tags = {"env": "test"}

        record5.merge({"expectations": {"relevance": 0.9}})

        assert record5.expectations == {"accuracy": 0.8, "relevance": 0.9}
        assert record5.tags == {"env": "test"}  # Unchanged

        record6 = SqlEvaluationDatasetRecord()
        record6.expectations = {"accuracy": 0.8}
        record6.tags = {"env": "test"}

        record6.merge({"tags": {"version": "1.0"}})

        assert record6.expectations == {"accuracy": 0.8}  # Unchanged
        assert record6.tags == {"env": "test", "version": "1.0"}


def test_sql_dataset_record_wrapping_unwrapping():
    from mlflow.entities.dataset_record import DATASET_RECORD_WRAPPED_OUTPUT_KEY

    entity = DatasetRecord(
        dataset_record_id="rec1",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs="string output",
        created_time=1000,
        last_update_time=1000,
    )

    sql_record = SqlEvaluationDatasetRecord.from_mlflow_entity(entity, "input_hash_123")

    assert sql_record.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: "string output"}

    unwrapped_entity = sql_record.to_mlflow_entity()
    assert unwrapped_entity.outputs == "string output"

    entity2 = DatasetRecord(
        dataset_record_id="rec2",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs=[1, 2, 3],
        created_time=1000,
        last_update_time=1000,
    )

    sql_record2 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity2, "input_hash_456")
    assert sql_record2.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: [1, 2, 3]}

    unwrapped_entity2 = sql_record2.to_mlflow_entity()
    assert unwrapped_entity2.outputs == [1, 2, 3]

    entity3 = DatasetRecord(
        dataset_record_id="rec3",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs=42,
        created_time=1000,
        last_update_time=1000,
    )

    sql_record3 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity3, "input_hash_789")
    assert sql_record3.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: 42}

    unwrapped_entity3 = sql_record3.to_mlflow_entity()
    assert unwrapped_entity3.outputs == 42

    entity4 = DatasetRecord(
        dataset_record_id="rec4",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs={"result": "answer"},
        created_time=1000,
        last_update_time=1000,
    )

    sql_record4 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity4, "input_hash_abc")
    assert sql_record4.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: {"result": "answer"}}

    unwrapped_entity4 = sql_record4.to_mlflow_entity()
    assert unwrapped_entity4.outputs == {"result": "answer"}

    entity5 = DatasetRecord(
        dataset_record_id="rec5",
        dataset_id="ds1",
        inputs={"q": "test"},
        outputs=None,
        created_time=1000,
        last_update_time=1000,
    )

    sql_record5 = SqlEvaluationDatasetRecord.from_mlflow_entity(entity5, "input_hash_def")
    assert sql_record5.outputs is None

    unwrapped_entity5 = sql_record5.to_mlflow_entity()
    assert unwrapped_entity5.outputs is None

    sql_record6 = SqlEvaluationDatasetRecord()
    sql_record6.outputs = {"old": "data"}

    sql_record6.merge({"outputs": "new string output"})
    assert sql_record6.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: "new string output"}

    sql_record7 = SqlEvaluationDatasetRecord()
    sql_record7.outputs = None

    sql_record7.merge({"outputs": {"new": "dict"}})
    assert sql_record7.outputs == {DATASET_RECORD_WRAPPED_OUTPUT_KEY: {"new": "dict"}}
