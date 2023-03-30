import uuid

from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
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


def test_creation_and_hydration():
    name = random_str()
    t1, t2 = 100, 150
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

    assert (
        str(model_version) == "<ModelVersion: aliases=[], creation_timestamp=12, "
        "current_stage='Archived', description='This is a test "
        "model.', last_updated_timestamp=100, "
        "name='myname', "
        "run_id='some run', run_link='http://localhost:5000/path/"
        "to/run', source='path/to/a/notebook', "
        "status='PENDING_REGISTRATION', status_message='Copying!', "
        "tags={}, user_id='user one', version='43'>"
    )
