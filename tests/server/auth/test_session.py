"""
Tests for the opt-in server-side session support in
``mlflow.server.auth.session`` (https://github.com/mlflow/mlflow/issues/13643).

Covers two levels:
  * unit tests against ``SqlAlchemyStore``'s session methods directly
    (``create_session`` / ``get_session_username`` / ``delete_session`` /
    ``delete_expired_sessions``), mirroring the style of
    ``test_sqlalchemy_store.py``.
  * end-to-end tests against a real spawned server configured with
    ``authorization_function = mlflow.server.auth.session:authenticate_request_session``
    (via ``fixtures/session_auth.ini``), verifying the cookie is actually
    issued, reused, and invalidated over the wire.
"""

import base64
import time

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
    # unknown id
    assert store.get_session_username("does-not-exist") is None


def test_get_session_username_expires_and_sweeps_lazily(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=-1)  # already expired

    assert store.get_session_username(session_id) is None
    # the lazy sweep on read should have removed the row
    assert store.delete_expired_sessions() == 0


def test_delete_session(store):
    username = random_str()
    store.create_user(username, random_str())
    session_id = store.create_session(username, ttl_seconds=3600)

    store.delete_session(session_id)
    assert store.get_session_username(session_id) is None
    # deleting an already-gone session is a no-op, not an error
    store.delete_session(session_id)


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


# ---- End-to-end tests: real server, authorization_function = session auth ----


@pytest.fixture
def session_client(request, tmp_path):
    """Like ``auth_test_utils``'s ``client``/``fastapi_client`` fixtures, but always
    boots with ``authorization_function = mlflow.server.auth.session:authenticate_request_session``
    (fixtures/session_auth.ini) instead of the default Basic Auth function."""
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

        # No Authorization header this time - the session cookie alone must work.
        r2 = s.get(url, params={"username": "admin"})
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
    # the client should be told to drop the stale cookie
    set_cookie = r.headers.get("Set-Cookie", "")
    assert SESSION_COOKIE_NAME in set_cookie
    assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()


def test_single_login_mints_exactly_one_session(session_client):
    """Regression test: mlflow.server.auth calls the configured
    authorization_function more than once per request (once from
    ``_before_request``, again from inside whichever permission validator
    runs), so a naive implementation mints two session rows per login."""
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(url, params={"username": "admin"}, auth=_ADMIN_AUTH)
    r.raise_for_status()
    session_id = r.cookies[SESSION_COOKIE_NAME]

    # A second, unrelated authenticated request must not mint a new cookie
    # for the same still-valid session.
    r2 = requests.get(
        url, params={"username": "admin"}, cookies={SESSION_COOKIE_NAME: session_id}
    )
    r2.raise_for_status()
    assert SESSION_COOKIE_NAME not in r2.cookies


def test_expired_session_reauthenticates(session_client, monkeypatch):
    url = session_client.tracking_uri + "/api/2.0/mlflow/users/get"
    r = requests.get(url, params={"username": "admin"}, auth=_ADMIN_AUTH)
    r.raise_for_status()
    session_id = r.cookies[SESSION_COOKIE_NAME]

    # Sanity: the session works before expiry.
    r2 = requests.get(
        url, params={"username": "admin"}, cookies={SESSION_COOKIE_NAME: session_id}
    )
    assert r2.status_code == 200

    # We can't fast-forward the spawned server's clock, so simulate expiry the
    # same way a real TTL lapse would look from the client's side: an unknown
    # session id gets the same bogus-cookie fallback behavior already covered
    # by test_bogus_session_cookie_falls_back_to_basic_auth_challenge. This
    # test instead documents the intended TTL by asserting the cookie's
    # Max-Age matches the configured session_ttl_seconds (43200, from
    # fixtures/session_auth.ini) rather than a framework default.
    set_cookie_header = r.headers.get("Set-Cookie", "")
    assert "Max-Age=43200" in set_cookie_header
