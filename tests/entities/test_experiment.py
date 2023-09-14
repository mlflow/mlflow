from mlflow.entities import Experiment, LifecycleStage
from mlflow.utils.time import get_current_time_millis

from tests.helper_functions import random_file, random_int


def _check(exp, exp_id, name, location, lifecyle_stage, creation_time, last_update_time):
    assert isinstance(exp, Experiment)
    assert exp.experiment_id == exp_id
    assert exp.name == name
    assert exp.artifact_location == location
    assert exp.lifecycle_stage == lifecyle_stage
    assert exp.creation_time == creation_time
    assert exp.last_update_time == last_update_time


def test_creation_and_hydration():
    exp_id = str(random_int())
    name = f"exp_{random_int()}_{random_int()}"
    lifecycle_stage = LifecycleStage.ACTIVE
    location = random_file(".json")
    creation_time = get_current_time_millis()
    last_update_time = get_current_time_millis()
    exp = Experiment(
        exp_id,
        name,
        location,
        lifecycle_stage,
        creation_time=creation_time,
        last_update_time=last_update_time,
    )
    _check(exp, exp_id, name, location, lifecycle_stage, creation_time, last_update_time)

    as_dict = {
        "experiment_id": exp_id,
        "name": name,
        "artifact_location": location,
        "lifecycle_stage": lifecycle_stage,
        "tags": {},
        "creation_time": creation_time,
        "last_update_time": last_update_time,
    }
    assert dict(exp) == as_dict
    proto = exp.to_proto()
    exp2 = Experiment.from_proto(proto)
    _check(exp2, exp_id, name, location, lifecycle_stage, creation_time, last_update_time)

    exp3 = Experiment.from_dictionary(as_dict)
    _check(exp3, exp_id, name, location, lifecycle_stage, creation_time, last_update_time)


def test_string_repr():
    exp = Experiment(
        experiment_id=0,
        name="myname",
        artifact_location="hi",
        lifecycle_stage=LifecycleStage.ACTIVE,
        creation_time=1662004217511,
        last_update_time=1662004217511,
    )
    assert (
        str(exp)
        == "<Experiment: artifact_location='hi', creation_time=1662004217511, experiment_id=0, "
        "last_update_time=1662004217511, lifecycle_stage='active', name='myname', tags={}>"
    )
