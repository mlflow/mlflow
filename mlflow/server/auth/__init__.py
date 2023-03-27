"""
Usage
-----

.. code-block:: bash

    mlflow server --app-name basic-auth
"""

import logging

from mlflow.server import app


_logger = logging.getLogger(__name__)


def enable_auth(app):
    _logger.warning(
        "This feature is still experimental and may change in a future release without warning"
    )

    @app.before_request
    def authenticate():
        # TODO: Implement authentication
        pass


enable_auth(app)
