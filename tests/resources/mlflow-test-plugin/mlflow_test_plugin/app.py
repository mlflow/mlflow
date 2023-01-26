import logging

# This would be all that plugin author is required to import
from mlflow.server import app as custom_app

# Can do custom logging on either the app or logging itself
# but you'll possibly have to clear the existing handlers or there will be duplicate output
# See https://docs.python.org/3/howto/logging-cookbook.html

app_logger = logging.getLogger(__name__)

# Configure the app
custom_app.config["MY_VAR"] = "config-var"


def is_logged_in():
    return False


@custom_app.before_request
def before_req_hook():
    """A custom before request handler.

    Can implement things such as authentication, special handling, etc.
    """
    app_logger.warning("Hello from before request!")
    if not is_logged_in():
        return "Unauthorized", 403
