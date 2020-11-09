import os
import pytest
import mlflow


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmpdir, request):
    from mlflow.utils.file_utils import path_to_local_file_uri

    tracking_uri = path_to_local_file_uri(os.path.join(tmpdir.strpath, "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    if "notrackingurimock" not in request.keywords:
        if "MLFLOW_TRACKING_URI" in os.environ:
            del os.environ["MLFLOW_TRACKING_URI"]
