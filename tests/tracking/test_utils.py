import mock
import os
import pytest

from mlflow.store.file_store import FileStore
from mlflow.store.rest_store import RestStore
from mlflow.tracking.utils import _get_store, _TRACKING_URI_ENV_VAR, _TRACKING_USERNAME_ENV_VAR, \
                                  _TRACKING_PASSWORD_ENV_VAR, _TRACKING_TOKEN_ENV_VAR, \
                                  _TRACKING_INSECURE_TLS_ENV_VAR


def test_get_store_file_store(tmpdir):
    env = {}
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, FileStore)
        assert store.root_directory == os.path.abspath("mlruns")

        # Make sure we look at the parameter...
        store = _get_store(tmpdir.strpath)
        assert isinstance(store, FileStore)
        assert store.root_directory == tmpdir


def test_get_store_basic_rest_store(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050"
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server:5050"
        assert store.get_host_creds().token is None


def test_get_store_rest_store_with_password(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_USERNAME_ENV_VAR: "Bob",
        _TRACKING_PASSWORD_ENV_VAR: "Ross",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server:5050"
        assert store.get_host_creds().username == "Bob"
        assert store.get_host_creds().password == "Ross"


def test_get_store_rest_store_with_token(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_TOKEN_ENV_VAR: "my-token",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().token == "my-token"


def test_get_store_rest_store_with_insecure(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_INSECURE_TLS_ENV_VAR: "true",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().ignore_tls_verification


def test_get_store_rest_store_with_no_insecure(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
        _TRACKING_INSECURE_TLS_ENV_VAR: "false",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert not store.get_host_creds().ignore_tls_verification

    # By default, should not ignore verification.
    env = {
        _TRACKING_URI_ENV_VAR: "https://my-tracking-server:5050",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert not store.get_host_creds().ignore_tls_verification


def test_get_store_databricks(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "databricks",
        'DATABRICKS_HOST': "https://my-tracking-server",
        'DATABRICKS_TOKEN': "abcdef",
    }
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        assert store.get_host_creds().host == "https://my-tracking-server"
        assert store.get_host_creds().token == "abcdef"


def test_get_store_databricks_profile(tmpdir):
    env = {
        _TRACKING_URI_ENV_VAR: "databricks://mycoolprofile",
    }
    # It's kind of annoying to setup a profile, and we're not really trying to test
    # that anyway, so just check if we raise a relevant exception.
    with mock.patch.dict(os.environ, env):
        store = _get_store()
        assert isinstance(store, RestStore)
        with pytest.raises(Exception) as e_info:
            store.get_host_creds()
        assert 'mycoolprofile' in str(e_info.value)
