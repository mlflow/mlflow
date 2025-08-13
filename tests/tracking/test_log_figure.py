import os
import posixpath
import uuid

import pytest

import mlflow
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.os import is_windows


@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_figure_matplotlib(subdir):
    import matplotlib.pyplot as plt

    filename = "figure.png"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])

    with mlflow.start_run():
        mlflow.log_figure(fig, artifact_file)
        plt.close(fig)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_figure_plotly_html(subdir):
    from plotly import graph_objects as go

    filename = "figure.html"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

    with mlflow.start_run():
        mlflow.log_figure(fig, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.skipif(is_windows, reason="https://github.com/plotly/Kaleido/issues/126")
@pytest.mark.parametrize("extension", ["png", "jpeg", "webp", "svg", "pdf"])
def test_log_figure_plotly_image(extension):
    from plotly import graph_objects as go

    subdir = "."
    filename = f"figure.{extension}"
    artifact_file = posixpath.join(subdir, filename)

    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

    with mlflow.start_run():
        mlflow.log_figure(fig, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


def test_log_figure_save_kwargs():
    from plotly import graph_objects as go

    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))
    with mlflow.start_run():
        name = "figure.html"
        div_id = uuid.uuid4().hex
        mlflow.log_figure(fig, name, save_kwargs={"div_id": div_id})
        artifact_uri = mlflow.get_artifact_uri(name)
        local_path = local_file_uri_to_path(artifact_uri)
        with open(local_path) as f:
            assert div_id in f.read()


@pytest.mark.parametrize("extension", ["", ".py"])
def test_log_figure_raises_error_for_unsupported_file_extension(extension):
    from plotly import graph_objects as go

    filename = f"figure{extension}"
    artifact_file = posixpath.join(".", filename)

    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

    with (
        mlflow.start_run(),
        pytest.raises(
            TypeError, match=f"Unsupported file extension for plotly figure: '{extension}'"
        ),
    ):
        mlflow.log_figure(fig, artifact_file)


def test_log_figure_raises_error_for_unsupported_figure_object_type():
    with mlflow.start_run(), pytest.raises(TypeError, match="Unsupported figure object type"):
        mlflow.log_figure("not_figure", "figure.png")
