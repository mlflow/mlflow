import os
import pytest
import mlflow


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
        if "notrackingurimock" not in request.keywords:
            del os.environ["MLFLOW_TRACKING_URI"]
