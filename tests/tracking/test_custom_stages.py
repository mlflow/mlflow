
"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""
from unittest import mock
import os
import sys
import pytest
import logging
import shutil
import tempfile
import json

from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import path_to_local_file_uri
from tests.tracking.integration_test_utils import _await_server_down_or_die, _init_server

# pylint: disable=unused-argument

# Root directory for all stores (backend or artifact stores) created during this suite
SUITE_ROOT_DIR = tempfile.mkdtemp("test_rest_tracking")
# Root directory for all artifact stores created during this suite
SUITE_ARTIFACT_ROOT_DIR = tempfile.mkdtemp(suffix="artifacts", dir=SUITE_ROOT_DIR)

CUSTOM_CONFIG_PATH = tempfile.NamedTemporaryFile(mode="w+", dir=SUITE_ROOT_DIR)
config_dict = {"model_stages": [{"name": name} for name in ["Staging", "Pre-Prod", "Custom Stage"]]}
json.dump(config_dict, CUSTOM_CONFIG_PATH)
CUSTOM_CONFIG_PATH.flush()

_logger = logging.getLogger(__name__)

def _get_sqlite_uri():
    path = path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "test-database.bd"))
    path = path[len("file://") :]

    # NB: It looks like windows and posix have different requirements on number of slashes for
    # whatever reason. Windows needs uri like 'sqlite:///C:/path/to/my/file' whereas posix expects
    # sqlite://///path/to/my/file
    prefix = "sqlite://" if sys.platform == "win32" else "sqlite:////"
    return prefix + path


# Backend store URIs to test against
BACKEND_URIS = [
    _get_sqlite_uri(),  # SqlAlchemy
]

# Map of backend URI to tuple (server URL, Process). We populate this map by constructing
# a server per backend URI
BACKEND_URI_TO_SERVER_URL_AND_PROC = {
    uri: _init_server(
        backend_uri=uri, 
        root_artifact_uri=SUITE_ARTIFACT_ROOT_DIR, 
        additional_env_variables={"_MLFLOW_CONFIG_PATH": CUSTOM_CONFIG_PATH.name}
    )
    for uri in BACKEND_URIS
}


def pytest_generate_tests(metafunc):
    """
    Automatically parametrize each each fixture/test that depends on `backend_store_uri` with the
    list of backend store URIs.
    """
    if "backend_store_uri" in metafunc.fixturenames:
        metafunc.parametrize("backend_store_uri", BACKEND_URIS)


@pytest.fixture(scope="module", autouse=True)
def server_urls():
    """
    Clean up all servers created for testing in `pytest_generate_tests`
    """
    yield
    for server_url, process in BACKEND_URI_TO_SERVER_URL_AND_PROC.values():
        _logger.info(f"Terminating server at {server_url}...")
        _logger.info(f"type = {type(process)}")
        process.terminate()
        _await_server_down_or_die(process)
    shutil.rmtree(SUITE_ROOT_DIR)


@pytest.fixture()
def tracking_server_uri(backend_store_uri):
    url, _ = BACKEND_URI_TO_SERVER_URL_AND_PROC[backend_store_uri]
    return url


@pytest.fixture()
def mlflow_client(tracking_server_uri):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    return mock.Mock(wraps=MlflowClient(tracking_server_uri))


def test_latest_models(mlflow_client, backend_store_uri):
    version_stage_mapping = (
        ("1", "Archived"),
        ("2", "Custom Stage"),
        ("3", "Archived"),
        ("4", "Custom Stage"),
        ("5", "Staging"),
        ("6", "Staging"),
        ("7", "None"),
        ("8", "Pre-Prod"),
        ("9", "Pre-Prod"),
    )
    name = "LatestVersionTest"
    mlflow_client.create_registered_model(name)

    for version, stage in version_stage_mapping:
        mv = mlflow_client.create_model_version(name, "path/to/model", "run_id")
        assert mv.version == version
        if stage != "None":
            mlflow_client.transition_model_version_stage(name, version, stage=stage)
        mvd = mlflow_client.get_model_version(name, version)
        assert mvd.current_stage == stage

    def get_latest(stages):
        latest = mlflow_client.get_latest_versions(name, stages)
        return {mvd.current_stage: mvd.version for mvd in latest}

    assert {"None": "7"} == get_latest(["None"])
    assert {"Staging": "6"} == get_latest(["Staging"])
    assert {"Pre-Prod": "9"} == get_latest(["Pre-Prod"])
    assert {"None": "7", "Staging": "6"} == get_latest(["None", "Staging"])
    assert {"Custom Stage": "4", "Staging": "6", "Archived": "3", "None": "7", "Pre-Prod": "9"} == get_latest(None)
    assert {"Custom Stage": "4", "Staging": "6", "Archived": "3", "None": "7", "Pre-Prod": "9"} == get_latest([])

