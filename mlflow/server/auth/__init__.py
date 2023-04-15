"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging
import uuid
import os
from pathlib import Path
from flask import Flask, request, make_response, Response, redirect, flash, render_template_string

from mlflow.server import app
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import _get_rest_path, catch_mlflow_exception
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
)

_AUTH_CONFIG_PATH_ENV_VAR = "MLFLOW_AUTH_CONFIG_PATH"

_logger = logging.getLogger(__name__)

store = SqlAlchemyStore()


class ROUTES:
    HOME = "/"
    USERS = _get_rest_path("/mlflow/users")
    SIGNUP = "/signup"


UNPROTECTED_ROUTES = [ROUTES.USERS, ROUTES.SIGNUP]


def is_unprotected_route(path: str) -> bool:
    if path.startswith(("/static", "/favicon.ico")):
        return True
    return path in UNPROTECTED_ROUTES


def make_basic_auth_response() -> Response:
    res = make_response()
    res.status_code = 401
    res.set_data(
        "You are not authenticated. Please set the environment variables "
        f"{_TRACKING_USERNAME_ENV_VAR} and {_TRACKING_PASSWORD_ENV_VAR}."
    )
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res


def _before_request():
    if is_unprotected_route(request.path):
        return

    _user = request.authorization.username if request.authorization else None
    _logger.debug(f"before_request: {request.method} {request.path} (user: {_user})")

    if request.authorization is None:
        return make_basic_auth_response()

    username = request.authorization.username
    password = request.authorization.password
    if not store.authenticate_user(username, password):
        # let user attempt login again
        return make_basic_auth_response()

    # TODO: Implement authorization
    pass


def _after_request(resp):
    # TODO: Implement post-request logic
    return resp


def signup():
    # TODO: add css
    return render_template_string(
        r"""
<form action="{{ users_route }}" method="post">
  Username:
  <br>
  <input type=text name=username>
  <br>
  Password:
  <br>
  <input type=password name=password>
  <br>
  <br>
  <input type="submit" value="Signup">
</form>
<style>
.alert.error {
  color: red;
}
</style>
{% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
    {% for category, message in messages %}
      <p class="alert {{ category }}">{{ message }}</p>
    {% endfor %}
  {% endif %}
{% endwith %}
""",
        users_route=ROUTES.USERS,
    )


@catch_mlflow_exception
def create_user():
    content_type = request.headers.get("Content-Type")
    if content_type == "application/x-www-form-urlencoded":
        username = request.form["username"]
        password = request.form["password"]
    elif content_type == "application/json":
        username = request.json["username"]
        password = request.json["password"]
    else:
        return make_response(f"Invalid content type: '{content_type}'", 400)

    if store.has_user(username):
        flash(f"Username has already been taken: '{username}'", category="error")
        return redirect(ROUTES.SIGNUP)

    store.create_user(username, password)
    flash(f"Successfully signed up user: '{username}'")
    return redirect(ROUTES.HOME)


def _get_auth_config_path():
    return os.environ.get(
        _AUTH_CONFIG_PATH_ENV_VAR, (Path(__file__).parent / "basic_auth.ini").resolve()
    )


def _enable_auth(app: Flask):
    """
    Enables authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    """
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )
    # secret key required for flashing
    if not app.secret_key:
        app.secret_key = str(uuid.uuid4())

    auth_config_path = _get_auth_config_path()
    auth_config = read_auth_config(auth_config_path)
    _logger.debug("Database URI: %s", auth_config.database_uri)
    store.init_db(auth_config.database_uri)

    app.add_url_rule(
        rule=ROUTES.SIGNUP,
        view_func=signup,
        methods=["GET"],
    )
    app.add_url_rule(
        rule=ROUTES.USERS,
        view_func=create_user,
        methods=["POST"],
    )

    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
