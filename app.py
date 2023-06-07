from fastapi import FastAPI
import os
import yaml
import logging

logger = logging.getLogger(__name__)

CONFIG_ENV_VAR = "CONFIG_PATH"


def create_route(s):
    async def route():
        return {"message": s}

    return route


def create_app():
    logger.warning("Creating app")
    with open(os.environ[CONFIG_ENV_VAR]) as f:
        config = yaml.safe_load(f)
    app = FastAPI()
    for s in config["routes"]:
        route = create_route(s)

        app.add_api_route(f"/{s}", route, methods=["GET"])

    return app
