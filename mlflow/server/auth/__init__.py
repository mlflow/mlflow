"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging

from mlflow.server import app


_logger = logging.getLogger(__name__)


def _before_request():
    # TODO: Implement authentication
    # TODO: Implement authorization
    pass


def _after_request(resp):
    # TODO: Implement post-request logic
    return resp


def _enable_auth(app):
    """
    Enables authentication and authorization for the MLflow server.

    :param app: The Flask app to enable authentication and authorization for.
    """
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )
    app.before_request(_before_request)
    app.after_request(_after_request)


_enable_auth(app)
