import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel, Extra, Field

CONFIG_PATH_ENV_VAR = "MLFLOW_GATEWAY_CONFIG_PATH"


logger = logging.getLogger(__name__)


class Config(BaseModel, extra=Extra.forbid):
    routes: List[str] = Field(...)


def validate_config(config: Dict[Any, Any]) -> Config:
    return Config(**config)


def create_route(s: str):
    async def route(request: Request):
        return s

    return route


def create_app():
    path = os.environ[CONFIG_PATH_ENV_VAR]
    print("Creating app")
    with open(path) as f:
        config = yaml.safe_load(f)

    valid_config_path = Path("validated.yml")
    try:
        config = validate_config(config)
        print("Validation succeeded")
    except Exception:
        logger.exception("Validation failed")
        if valid_config_path.exists():
            with valid_config_path.open() as f:
                config = validate_config(yaml.safe_load(f))
        else:
            raise
    else:
        print("Saving validated config")
        with valid_config_path.open("w") as f:
            yaml.safe_dump(config.dict(), f)

    print("Config:", config)
    app = FastAPI()
    for route in config.routes:
        r = create_route(route)
        app.add_api_route("/" + route, r, methods=["GET"])

    return app
