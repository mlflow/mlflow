import pytest
import random
import uuid

from tests.helper_functions import random_str, random_int

from mlflow.entities import RunInfo, LifecycleStage, RunStatus
from mlflow.entities import Metric, RunData, Param, RunTag
from mlflow.utils.time_utils import get_current_time_millis


@pytest.fixture(scope="module")
def test_run_data():
    metrics = [
        Metric(
            key=random_str(10),
            value=random_int(0, 1000),
            timestamp=get_current_time_millis() + random_int(-1e4, 1e4),
            step=random_int(),
        )
    ]
    params = [Param(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]
    tags = [RunTag(random_str(10), random_str(random_int(10, 35))) for _ in range(10)]

    rd = RunData(metrics=metrics, params=params, tags=tags)

    return rd, metrics, params, tags


@pytest.fixture(scope="module")
def test_run_info():
    run_id = str(uuid.uuid4())
    experiment_id = str(random_int(10, 2000))
    user_id = random_str(random_int(10, 25))
    run_name = random_str(random_int(10, 25))
    status = RunStatus.to_string(random.choice(RunStatus.all_status()))
    start_time = random_int(1, 10)
    end_time = start_time + random_int(1, 10)
    lifecycle_stage = LifecycleStage.ACTIVE
    artifact_uri = random_str(random_int(10, 40))
    ri = RunInfo(
        run_uuid=run_id,
        run_id=run_id,
        run_name=run_name,
        experiment_id=experiment_id,
        user_id=user_id,
        status=status,
        start_time=start_time,
        end_time=end_time,
        lifecycle_stage=lifecycle_stage,
        artifact_uri=artifact_uri,
    )
    return (
        ri,
        run_id,
        run_name,
        experiment_id,
        user_id,
        status,
        start_time,
        end_time,
        lifecycle_stage,
        artifact_uri,
    )
