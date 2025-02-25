"""
To run a tracking server with this app, use `mlflow server --app-name custom_app`.
"""

import logging

# This would be all that plugin author is required to import
from mlflow.server import app as custom_app

# Can do custom logging on either the app or logging itself
# but you'll possibly have to clear the existing handlers or there will be duplicate output
# See https://docs.python.org/3/howto/logging-cookbook.html

app_logger = logging.getLogger(__name__)

# Configure the app
custom_app.config["MY_VAR"] = "config-var"
app_logger.warning(f"Using {__name__}")


def is_logged_in():
    return True


@custom_app.before_request
def before_req_hook():
    """A custom before request handler.

    Can implement things such as authentication, special handling, etc.
    """
    if not is_logged_in():
        app_logger.warning("Hello from before request!")
        return "Unauthorized", 403


@custom_app.route("/custom/endpoint", methods=["GET"])
def custom_endpoint():
    """A custom endpoint."""
    return "custom_endpoint", 200
