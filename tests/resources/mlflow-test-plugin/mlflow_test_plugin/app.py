"""
To run a tracking server with this app, use `mlflow server --app-name custom_app`.
"""

import logging

from mlflow.server.fastapi_app import create_fastapi_app

app_logger = logging.getLogger(__name__)
app_logger.warning(f"Using {__name__}")

custom_app = create_fastapi_app()


def is_logged_in():
    return True


@custom_app.get("/custom/endpoint")
async def custom_endpoint():
    return "custom_endpoint"
