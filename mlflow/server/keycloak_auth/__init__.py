import logging
import uuid


from flask import Flask, Response, make_response, request
from mlflow.server.keycloak_auth.config import read_auth_config

from mlflow.server.keycloak_auth.constants import INGESTION_CLIENT_READ_ROLE, INSUFFICIENT_PERMISSION_BY_ADMIN, TOKEN_NOT_FOUND, TOKEN_NOT_VALID
from mlflow.server import app
from werkzeug.exceptions import Unauthorized
import httpx
import jwt
import json

_logger = logging.getLogger(__name__)

auth_config = read_auth_config()

def is_unprotected_route(path: str) -> bool:
    return path.startswith(("/static", "/favicon.ico", "/health"))


def _before_request():

    if is_unprotected_route(request.path):
        return

    validate_token()

@app.errorhandler(Unauthorized)
def make_basic_auth_response(unauthorized_exception) -> Response:
    errorMessages = {"errorMessages": unauthorized_exception.description }

    res = make_response(json.dumps(errorMessages))
    res.status_code = 401
    res.content_type = "application/json"
    return res

def validate_token():
    bearer_token = request.headers.get("Authorization")

    if not bearer_token:
        _logger.error(TOKEN_NOT_FOUND)
        raise Unauthorized(description = TOKEN_NOT_FOUND)

    KEYCLOAK_HOST = auth_config.host
    REALM_NAME = auth_config.realm_name

    validate_token_url = (
        f"""{KEYCLOAK_HOST}realms/{REALM_NAME}/protocol/openid-connect/userinfo"""
    )
    headers = {"Authorization": bearer_token}

    try:
        with httpx.Client() as client:
            response =  client.get(validate_token_url, headers=headers)
            response.raise_for_status()
            user_info = response.json()

        claims_keys = ["sub", "email", "given_name"]
        for key in claims_keys:
            if key not in user_info:
                _logger.error(f"Claim not found in token: {key}")
                raise Unauthorized(description = TOKEN_NOT_VALID)

        authorize_ingestion_role(bearer_token)

    except httpx.HTTPStatusError as err:
        _logger.error(f"Token validation failed: {err}")
        raise Unauthorized(description = TOKEN_NOT_VALID)
    except Exception as ex:
        _logger.error(f"Token validation failed: {ex}")
        raise Unauthorized(str(ex))

def authorize_ingestion_role(bearer_token):

    token = bearer_token.split()[1]

    decoded_token = jwt.decode(token, options={"verify_signature": False})

    roles = decoded_token.get("realm_access").get("roles")

    if INGESTION_CLIENT_READ_ROLE not in roles:
        raise Unauthorized(detail = INSUFFICIENT_PERMISSION_BY_ADMIN)

def create_app(app: Flask = app):
    """
    A factory to enable authentication for the MLflow server using keycloak.

    Args:
        app: The Flask app to enable authentication for.

    Returns:
        The app with authentication enabled.
    """

    # secret key required for flashing
    if not app.secret_key:
        app.secret_key = str(uuid.uuid4())

    app.before_request(_before_request)

    return app
