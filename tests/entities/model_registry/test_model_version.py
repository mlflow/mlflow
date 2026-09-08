import uuid

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_deployment_job_state import (
    ModelVersionDeploymentJobState,
)
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str


def _check(
    model_version,
    name,
    version,
    creation_timestamp,
    last_updated_timestamp,
    description,
    user_id,
    current_stage,
    source,
    run_id,
    status,
    status_message,
    tags,
    aliases,
    workspace=DEFAULT_WORKSPACE_NAME,
):
    assert isinstance(model_version, ModelVersion)
    assert model_version.name == name
    assert model_version.version == version
    assert model_version.creation_timestamp == creation_timestamp
    assert model_version.last_updated_timestamp == last_updated_timestamp
    assert model_version.description == description
    assert model_version.user_id == user_id
    assert model_version.current_stage == current_stage
    assert model_version.source == source
    assert model_version.run_id == run_id
    assert model_version.status == status
    assert model_version.status_message == status_message
    assert model_version.tags == tags
    assert model_version.aliases == aliases
    assert model_version.workspace == workspace


def test_creation_and_hydration():
    name = random_str()
    t1 = 100
    t2 = 150
    source = "path/to/source"
    run_id = uuid.uuid4().hex
    run_link = "http://localhost:5000/path/to/run"
    tags = [ModelVersionTag("key", "value"), ModelVersionTag("randomKey", "not a random value")]
    aliases = ["test_alias"]
    mvd = ModelVersion(
        name,
        "5",
        t1,
        t2,
        "version five",
        "user 1",
        "Production",
        source,
        run_id,
        "READY",
        "Model version #5 is ready to use.",
        tags,
        run_link,
        aliases,
    )
    _check(
        mvd,
        name,
        "5",
        t1,
        t2,
        "version five",
        "user 1",
        "Production",
        source,
        run_id,
        "READY",
        "Model version #5 is ready to use.",
        {tag.key: tag.value for tag in (tags or [])},
        ["test_alias"],
    )

    expected_dict = {
        "name": name,
        "version": "5",
        "creation_timestamp": t1,
        "last_updated_timestamp": t2,
        "description": "version five",
        "user_id": "user 1",
        "current_stage": "Production",
        "source": source,
        "run_id": run_id,
        "run_link": run_link,
        "status": "READY",
        "status_message": "Model version #5 is ready to use.",
        "tags": {tag.key: tag.value for tag in (tags or [])},
        "aliases": ["test_alias"],
        "model_id": None,
        "metrics": None,
        "params": None,
        "deployment_job_state": None,
        "workspace": DEFAULT_WORKSPACE_NAME,
    }
    model_version_as_dict = dict(mvd)
    assert model_version_as_dict == expected_dict

    proto = mvd.to_proto()
    assert proto.name == name
    assert proto.version == "5"
    assert proto.status == ModelVersionStatus.from_string("READY")
    assert proto.status_message == "Model version #5 is ready to use."
    assert {tag.key for tag in proto.tags} == {"key", "randomKey"}
    assert {tag.value for tag in proto.tags} == {"value", "not a random value"}
    assert proto.aliases == ["test_alias"]
    mvd_2 = ModelVersion.from_proto(proto)
    _check(
        mvd_2,
        name,
        "5",
        t1,
        t2,
        "version five",
        "user 1",
        "Production",
        source,
        run_id,
        "READY",
        "Model version #5 is ready to use.",
        {tag.key: tag.value for tag in (tags or [])},
        ["test_alias"],
    )

    expected_dict.update({"registered_model": RegisteredModel(name)})
    expected_dict["tags"] = tags
    mvd_3 = ModelVersion.from_dictionary(expected_dict)
    _check(
        mvd_3,
        name,
        "5",
        t1,
        t2,
        "version five",
        "user 1",
        "Production",
        source,
        run_id,
        "READY",
        "Model version #5 is ready to use.",
        {tag.key: tag.value for tag in (tags or [])},
        ["test_alias"],
    )


def test_string_repr():
    model_version = ModelVersion(
        name="myname",
        version="43",
        creation_timestamp=12,
        last_updated_timestamp=100,
        description="This is a test model.",
        user_id="user one",
        current_stage="Archived",
        source="path/to/a/notebook",
        run_id="some run",
        run_link="http://localhost:5000/path/to/run",
        status="PENDING_REGISTRATION",
        status_message="Copying!",
        tags=[],
        aliases=[],
    )

    assert str(model_version) == (
        "<ModelVersion: aliases=[], creation_timestamp=12, current_stage='Archived', "
        "deployment_job_state=None, "
        "description='This is a test model.', last_updated_timestamp=100, metrics=None, "
        "model_id=None, name='myname', params=None, run_id='some run', "
        "run_link='http://localhost:5000/path/to/run', source='path/to/a/notebook', "
        "status='PENDING_REGISTRATION', status_message='Copying!', tags={}, user_id='user one', "
        "version='43', workspace='default'>"
    )


def test_model_version_non_default_workspace_round_trip():
    workspace = f"team-{random_str()}"
    run_id = uuid.uuid4().hex
    model_version = ModelVersion(
        name="roundtrip-model",
        version="7",
        creation_timestamp=10,
        last_updated_timestamp=20,
        description="non-default workspace",
        user_id="user-10",
        current_stage="Production",
        source="path/to/model",
        run_id=run_id,
        status="READY",
        status_message="ready",
        tags=[],
        aliases=[],
        workspace=workspace,
    )

    as_dict = dict(model_version)
    assert as_dict["workspace"] == workspace
    as_dict["tags"] = []

    hydrated = ModelVersion.from_dictionary(as_dict)
    assert hydrated.workspace == workspace
    assert workspace in str(hydrated)


def test_model_version_deployment_job_state_proto_round_trip():
    deployment_job_state = ModelVersionDeploymentJobState(
        job_id="job-123",
        run_id="run-456",
        job_state="CONNECTED",
        run_state="RUNNING",
        current_task_name="Evaluation",
    )
    model_version = ModelVersion(
        name="deployment-job-model",
        version="1",
        creation_timestamp=1,
        deployment_job_state=deployment_job_state,
    )

    proto = model_version.to_proto()
    assert proto.HasField("deployment_job_state")
    assert proto.deployment_job_state.job_id == "job-123"
    assert proto.deployment_job_state.run_id == "run-456"
    assert proto.deployment_job_state.current_task_name == "Evaluation"

    hydrated = ModelVersion.from_proto(proto)
    assert hydrated.deployment_job_state == deployment_job_state


def test_model_version_without_deployment_job_state_round_trip():
    model_version = ModelVersion(name="deployment-job-model", version="1", creation_timestamp=1)

    proto = model_version.to_proto()
    assert not proto.HasField("deployment_job_state")
    assert ModelVersion.from_proto(proto).deployment_job_state is None
