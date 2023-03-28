"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging

from flask import Flask, request, make_response, Response

from mlflow.server import app
from mlflow.server.auth import store
from mlflow.server.auth.config import app_config
from mlflow.server.handlers import get_endpoints


_logger = logging.getLogger(__name__)


PROTECTED_ROUTES = []


def is_protected_route(path: str) -> bool:
    if path.startswith("/static-files"):
        return True
    return path in PROTECTED_ROUTES


def make_forbidden_response() -> Response:
    res = make_response("Permission denied")
    res.status_code = 403
    return res


def make_invalid_login_response() -> Response:
    res = make_response("Invalid username or password")
    res.status_code = 403
    return res


def make_basic_auth_response() -> Response:
    res = make_response()
    res.status_code = 401
    res.headers["WWW-Authenticate"] = 'Basic realm="mlflow"'
    return res


def validate_permission(path, username):
    # TODO
    pass


def signup():
    # TODO: add css
    return """
<form action="/users" method="post">
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
"""


def _before_request():
    # TODO: Implement authentication
    # TODO: Implement authorization
    # TODO: remove
    _logger.info("before_request: %s %s %s", request.method, request.path, request.authorization)
    if not is_protected_route(request.path):
        return

    if request.authorization is None:
        return make_basic_auth_response()

    username = request.authorization.username
    password = request.authorization.password
    if not store.authenticate_user(username, password):
        return make_invalid_login_response()

    # authorization
    validate_permission(request.path, username)


def _after_request(resp):
    # TODO: Implement post-request logic
    return resp


def _enable_auth(app: Flask):
    """
    Enables authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    """
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )
    # TODO: remove
    _logger.info("Database URI: %s", app_config.database_uri)
    app.config["SQLALCHEMY_DATABASE_URI"] = app_config.database_uri

    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
