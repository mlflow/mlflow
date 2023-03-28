"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging

from flask import Flask, request

from mlflow.server import app
from mlflow.server.auth.config import app_config


_logger = logging.getLogger(__name__)


PROTECTED_ROUTES = []


def authenticate_user(username: str, password: str) -> bool:
    # TODO
    pass


def invalid_login():
    # TODO
    pass


def verify_permission(path, username):
    # TODO
    pass


def _before_request():
    # TODO: Implement authentication
    # TODO: Implement authorization
    # TODO: remove
    _logger.info("before_request: %s %s %s", request.method, request.path, request.authorization)
    if request.path not in PROTECTED_ROUTES:
        return

    if request.authorization is not None:
        ...

    username = request.authorization.username
    password = request.authorization.password
    if not authenticate_user(username, password):
        return invalid_login()

    # authorization
    verify_permission(request.path, username)


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
