"""
Tests for mlflow.server.auth.session (https://github.com/mlflow/mlflow/issues/13643):
store-level unit tests, plus end-to-end tests against a spawned server
configured with fixtures/session_auth.ini.
"""

import pytest
import requests

from mlflow.server.auth.session import SESSION_COOKIE_NAME

from tests.helper_functions import random_str

pytestmark = pytest.mark.notrackingurimock


# ---- Unit tests: SqlAlchemyStore session methods ----


@pytest.fixture
def store(tmp_sqlite_uri):
    from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

    store = SqlAlchemyStore()
    store.init_db(tmp_sqlite_uri)
    return store


def test_create_and_get_session(store):
    username = random_str()
    store.create_user(username, random_str())

    session_id = store.create_session(username, ttl_seconds=3600)
    assert isinstance(session_id, str) and session_id

    assert store.get_session_username(session_id) == username
    assert store.get_session_username("does-not-exist") is None


def test_get_session_username_expires_and_sweeps_lazily(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=-1)  # already expired

    assert store.get_session_username(session_id) is None
    assert store.delete_expired_sessions() == 0  # already swept on read


def test_delete_session(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=3600)

    store.delete_session(session_id)
    assert store.get_session_username(session_id) is None
    store.delete_session(session_id)  # no-op, not an error


def test_delete_expired_sessions_sweeps_only_expired(store):
    username = random_str()
    store.create_user(username, random_str())
    live = store.create_session(username, ttl_seconds=3600)
    expired1 = store.create_session(username, ttl_seconds=-10)
    expired2 = store.create_session(username, ttl_seconds=-1)

    assert store.delete_expired_sessions() == 2
    assert store.get_session_username(live) == username
    assert store.get_session_username(expired1) is None
    assert store.get_session_username(expired2) is None


def test_delete_user_invalidates_their_sessions(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=3600)

    store.delete_user(username)
    assert store.get_session_username(session_id) is None


def test_delete_user_does_not_invalidate_other_users_sessions(store):
    username1 = random_str()
    username2 = random_str()
    store.create_user(username1, random_str())
    store.create_user(username2, random_str())
    session1 = store.create_session(username1, ttl_seconds=3600)
    session2 = store.create_session(username2, ttl_seconds=3600)

    store.delete_user(username1)
    assert store.get_session_username(session1) is None
    assert store.get_session_username(session2) == username2


def test_update_user_password_invalidates_sessions(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=3600)

    store.update_user(username, password=random_str())
    assert store.get_session_username(session_id) is None


def test_update_user_is_admin_only_does_not_invalidate_sessions(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=3600)

    store.update_user(username, is_admin=True)
    assert store.get_session_username(session_id) == username


# ---- End-to-end tests: real server, authorization_function = session auth ----


@pytest.fixture
def session_client(request, tmp_path):
    """Like the ``client``/``fastapi_client`` fixtures in test_auth.py, but boots
    with fixtures/session_auth.ini (authorization_function = session auth)."""
    from mlflow import MlflowClient
    from mlflow.environment_variables import MLFLOW_FLASK_SERVER_SECRET_KEY
    from mlflow.utils.os import is_windows

    from tests.server.auth.test_auth import _isolate_auth_config
    from tests.tracking.integration_test_utils import _init_server

    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    extra_env = _isolate_auth_config(
        {"MLFLOW_AUTH_CONFIG_PATH": "fixtures/session_auth.ini"}, tmp_path
    )
    extra_env[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = "my-secret-key"

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        yield MlflowClient(url)


_ADMIN_AUTH = ("admin", "password1234")


def test_basic_auth_login_issues_session_cookie(session_client):
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(url, params={"username": "admin"}, auth=_ADMIN_AUTH)
    r.raise_for_status()
    assert SESSION_COOKIE_NAME in r.cookies
    assert r.cookies[SESSION_COOKIE_NAME]


def test_session_cookie_authenticates_without_credentials(session_client):
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    with requests.Session() as s:
        r = s.get(url, params={"username": "admin"}, auth=_ADMIN_AUTH)
        r.raise_for_status()
        assert SESSION_COOKIE_NAME in s.cookies

        r2 = s.get(url, params={"username": "admin"})  # no Authorization header
        r2.raise_for_status()
        assert r2.json()["user"]["username"] == "admin"


def test_missing_credentials_and_cookie_is_unauthenticated(session_client):
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(url, params={"username": "admin"})
    assert r.status_code == 401
    assert "Basic" in r.headers.get("WWW-Authenticate", "")


def test_bogus_session_cookie_falls_back_to_basic_auth_challenge(session_client):
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(
        url,
        params={"username": "admin"},
        cookies={SESSION_COOKIE_NAME: "not-a-real-session-id"},
    )
    assert r.status_code == 401
    set_cookie = r.headers.get("Set-Cookie", "")  # stale cookie should be cleared
    assert SESSION_COOKIE_NAME in set_cookie
    assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()


def test_single_login_mints_exactly_one_session(session_client):
    """Regression test: authorization_function is called more than once per
    request (_before_request, then again per validator), which used to mint
    two session rows per login."""
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(url, params={"username": "admin"}, auth=_ADMIN_AUTH)
    r.raise_for_status()
    session_id = r.cookies[SESSION_COOKIE_NAME]

    r2 = requests.get(
        url, params={"username": "admin"}, cookies={SESSION_COOKIE_NAME: session_id}
    )
    r2.raise_for_status()
    assert SESSION_COOKIE_NAME not in r2.cookies


def test_session_ttl_matches_config(session_client):
    # Can't fast-forward the spawned server's clock to test real expiry (that's
    # covered indirectly by test_bogus_session_cookie_falls_back_to_basic_auth_challenge);
    # instead just confirm the cookie's Max-Age matches session_ttl_seconds
    # from fixtures/session_auth.ini rather than some framework default.
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(url, params={"username": "admin"}, auth=_ADMIN_AUTH)
    r.raise_for_status()
    assert "Max-Age=43200" in r.headers.get("Set-Cookie", "")


def test_explicit_basic_auth_overrides_session_cookie(session_client):
    """Regression test: a session cookie must not silently override explicit
    Basic Auth credentials for a different user."""
    from tests.server.auth.auth_test_utils import create_user

    username_a, password_a = create_user(session_client.tracking_uri)
    username_b, password_b = create_user(session_client.tracking_uri)

    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"

    r = requests.get(url, params={"username": username_a}, auth=(username_a, password_a))
    r.raise_for_status()
    session_a = r.cookies[SESSION_COOKIE_NAME]

    # Send username_b's own credentials alongside username_a's cookie: the
    # explicit credentials must win, so this must authenticate as username_b
    # and be allowed to read its own record.
    r2 = requests.get(
        url,
        params={"username": username_b},
        auth=(username_b, password_b),
        cookies={SESSION_COOKIE_NAME: session_a},
    )
    assert r2.status_code == 200
    assert r2.json()["user"]["username"] == username_b
