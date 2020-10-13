import pytest
import numpy as np
import matplotlib.pyplot as plt
import mlflow.shap


def yield_artifacts(run_id, path=None):
    """
    Yields all artifacts in the specified run.
    """
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


@pytest.mark.large
@pytest.mark.parametrize("np_obj", [np.float(0.0), np.array([0.0])])
def test_log_numpy(np_obj):

    with mlflow.start_run() as run:
        mlflow.shap._log_numpy(np_obj, "test.npy")
        mlflow.shap._log_numpy(np_obj, "test.npy", artifact_path="dir")

    artifacts = set(yield_artifacts(run.info.run_id))
    assert artifacts == {"test.npy", "dir/test.npy"}


@pytest.mark.large
def test_log_matplotlib_figure():

    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])

    with mlflow.start_run() as run:
        mlflow.shap._log_matplotlib_figure(fig, "test.png")
        mlflow.shap._log_matplotlib_figure(fig, "test.png", artifact_path="dir")

    artifacts = set(yield_artifacts(run.info.run_id))
    assert artifacts == {"test.png", "dir/test.png"}
