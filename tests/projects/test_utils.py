import mock

from mlflow import tracking
from mlflow.entities import run, run_status
from mlflow.projects import utils


def test_gen_envvars_from_run():
    run_info = run.RunInfo(run_uuid="my_run", experiment_id=23, user_id="j.doe",
                           status=run_status.RunStatus.RUNNING, start_time=0,
                           end_time=1, lifecycle_stage=None)
    current_run = run.Run(run_info=run_info, run_data=None)
    with mock.patch('mlflow.tracking.get_tracking_uri', return_value="https://my_mlflow:5000"):
        assert {
            tracking._RUN_ID_ENV_VAR: "my_run",
            tracking._TRACKING_URI_ENV_VAR: "https://my_mlflow:5000",
            tracking._EXPERIMENT_ID_ENV_VAR: "23",
        } == utils.generate_env_vars_to_attach_to_run(current_run)
