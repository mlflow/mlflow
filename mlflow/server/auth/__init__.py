"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging

from mlflow.server import app


_logger = logging.getLogger(__name__)


def before_request():
    # TODO: Implement authentication
    # TODO: Implement authorization
    pass


def after_request(resp):
    # TODO: Implement post-request logic
    return resp


def enable_auth(app):
    """
    Enables authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    """
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )
    app.before_request(before_request)
    app.after_request(after_request)


enable_auth(app)
