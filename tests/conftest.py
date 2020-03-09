import os

import pytest

import mlflow
from mlflow.utils.file_utils import path_to_local_sqlite_uri


@pytest.fixture
def reset_mock():
    cache = []

    def set_mock(obj, attr, mock):
        cache.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, mock)

    yield set_mock

    for obj, attr, value in cache:
        setattr(obj, attr, value)
    cache[:] = []


@pytest.fixture(autouse=True)
def tracking_uri_mock(tmpdir, request):
    try:
        if 'notrackingurimock' not in request.keywords:
            mlflow.set_tracking_uri(path_to_local_sqlite_uri(
                os.path.join(tmpdir.strpath, 'mlruns')))
        yield tmpdir
    finally:
        mlflow.set_tracking_uri(None)
