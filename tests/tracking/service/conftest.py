import os
import pytest
import mlflow
from mlflow.tracking.fluent import _active_experiment_id


@pytest.fixture(autouse=True)
@pytest.mark.usefixtures("reset_active_experiment")
def tracking_uri_mock(tmpdir, request):
    try:
        from mlflow.utils.file_utils import path_to_local_file_uri

        if "notrackingurimock" not in request.keywords:
            tracking_uri = path_to_local_file_uri(os.path.join(tmpdir.strpath, "mlruns"))
            mlflow.set_tracking_uri(tracking_uri)
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        yield tmpdir
    finally:
        mlflow.set_tracking_uri(None)
        global _active_experiment_id
        _active_experiment_id = None
        if "notrackingurimock" not in request.keywords:
            del os.environ["MLFLOW_TRACKING_URI"]
