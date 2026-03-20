from mlflow.entities import Experiment, ExperimentTag, LifecycleStage
from mlflow.entities.trace_location import UnityCatalog
from mlflow.utils.mlflow_tags import (
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_ANNOTATIONS_TABLE,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_LOG_STORAGE_TABLE,
    MLFLOW_EXPERIMENT_DATABRICKS_TRACE_SPAN_STORAGE_TABLE,
)
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_file, random_int


def _check(
    exp,
    exp_id,
    name,
    location,
    lifecycle_stage,
    creation_time,
    last_update_time,
    workspace,
):
    assert isinstance(exp, Experiment)
    assert exp.experiment_id == exp_id
    assert exp.name == name
    assert exp.artifact_location == location
    assert exp.lifecycle_stage == lifecycle_stage
    assert exp.creation_time == creation_time
    assert exp.last_update_time == last_update_time
    assert exp.workspace == workspace


def test_creation_and_hydration():
    exp_id = str(random_int())
    name = f"exp_{random_int()}_{random_int()}"
    lifecycle_stage = LifecycleStage.ACTIVE
    location = random_file(".json")
    creation_time = get_current_time_millis()
    last_update_time = get_current_time_millis()
    expected_workspace = DEFAULT_WORKSPACE_NAME
    exp = Experiment(
        exp_id,
        name,
        location,
        lifecycle_stage,
        creation_time=creation_time,
        last_update_time=last_update_time,
    )
    _check(
        exp,
        exp_id,
        name,
        location,
        lifecycle_stage,
        creation_time,
        last_update_time,
        expected_workspace,
    )

    as_dict = {
        "experiment_id": exp_id,
        "name": name,
        "artifact_location": location,
        "lifecycle_stage": lifecycle_stage,
        "tags": {},
        "creation_time": creation_time,
        "last_update_time": last_update_time,
        "trace_location": None,
        "workspace": expected_workspace,
    }
    assert dict(exp) == as_dict
    proto = exp.to_proto()
    exp2 = Experiment.from_proto(proto)
    _check(
        exp2,
        exp_id,
        name,
        location,
        lifecycle_stage,
        creation_time,
        last_update_time,
        expected_workspace,
    )

    exp3 = Experiment.from_dictionary(as_dict)
    _check(
        exp3,
        exp_id,
        name,
        location,
        lifecycle_stage,
        creation_time,
        last_update_time,
        expected_workspace,
    )


def test_string_repr():
    exp = Experiment(
        experiment_id=0,
        name="myname",
        artifact_location="hi",
        lifecycle_stage=LifecycleStage.ACTIVE,
        creation_time=1662004217511,
        last_update_time=1662004217511,
    )
    assert str(exp) == (
        "<Experiment: artifact_location='hi', creation_time=1662004217511, "
        "experiment_id=0, last_update_time=1662004217511, "
        "lifecycle_stage='active', name='myname', tags={}, "
        "trace_location=None, workspace='default'>"
    )


def test_trace_location_lazy_resolves_from_tags():
    exp = Experiment(
        experiment_id="1",
        name="test",
        artifact_location="/tmp",
        lifecycle_stage=LifecycleStage.ACTIVE,
        tags=[
            ExperimentTag(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH, "cat.sch.pfx"),
            ExperimentTag(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_SPAN_STORAGE_TABLE, "cat.sch.spans"),
            ExperimentTag(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_LOG_STORAGE_TABLE, "cat.sch.logs"),
            ExperimentTag(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_ANNOTATIONS_TABLE, "cat.sch.annot"),
        ],
    )
    loc = exp.trace_location
    assert isinstance(loc, UnityCatalog)
    assert loc.catalog_name == "cat"
    assert loc.schema_name == "sch"
    assert loc.table_prefix == "pfx"
    assert loc._otel_spans_table_name == "cat.sch.spans"
    assert loc._otel_logs_table_name == "cat.sch.logs"
    assert loc._annotations_table_name == "cat.sch.annot"

    # Second access returns the cached instance
    assert exp.trace_location is loc


def test_trace_location_none_without_tags():
    exp = Experiment(
        experiment_id="1",
        name="test",
        artifact_location="/tmp",
        lifecycle_stage=LifecycleStage.ACTIVE,
    )
    assert exp.trace_location is None


def test_trace_location_setter_overrides_lazy():
    exp = Experiment(
        experiment_id="1",
        name="test",
        artifact_location="/tmp",
        lifecycle_stage=LifecycleStage.ACTIVE,
        tags=[
            ExperimentTag(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH, "cat.sch.pfx"),
        ],
    )
    override = UnityCatalog("other_cat", "other_sch", "other_pfx")
    exp.trace_location = override
    assert exp.trace_location is override
