"""Sample JWT authentication module for testing purposes.

NOT SUITABLE FOR PRODUCTION USE.
"""

import logging
from typing import Union

import uuid
import jwt
from flask import Response, make_response, request
from werkzeug.datastructures import Authorization

from mlflow.server.auth.routes import (
    GET_USER,
    CREATE_USER
)
from mlflow.server.auth.entities import User
from mlflow.server.auth.config import read_auth_config
from mlflow.utils.rest_utils import http_request_safe, MlflowHostCreds

auth_config = read_auth_config()



BEARER_PREFIX = "bearer "
_logger = logging.getLogger(__name__)
host_creds = MlflowHostCreds(host="http://localhost:5000", username=auth_config.admin_username, password=auth_config.admin_password)

def user_exists(username):
    resp = http_request_safe(host_creds, GET_USER, "GET", params={"username": username })
    return resp.status_code == 200

def create_user(username):
    resp = http_request_safe(host_creds, CREATE_USER, "POST", json={"username": username, "password": uuid.uuid4().hex})
    return User.from_json(resp["user"])


def authenticate_sso_request() -> Union[Authorization, Response]:
    _logger.debug("Getting token")
    error_response = make_response()
    error_response.status_code = 401
    error_response.set_data(
        "You are not authenticated. Please provide a valid JWT Bearer token with the request."
    )
    error_response.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'

    token = request.headers.get("authorization")

    if token is None or token.lower().startswith(BEARER_PREFIX):
        token = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJPZHhRNjdVMlFHekh4Y0ExLXp3c0tITVgtUTdnZGF0bnF0WXJQRGp4NUh3In0.eyJleHAiOjE3MzQ2MTkzMTgsImlhdCI6MTczNDYxOTAxOCwiYXV0aF90aW1lIjoxNzM0NjA5ODA3LCJqdGkiOiIzZGEwMzUzMC05ZjBhLTQwMjEtYjNmYS1iZDZhMDQwYzY5ZTAiLCJpc3MiOiJodHRwczovL2F1dGgua29zbW9zbWwud2lwL3JlYWxtcy90ZWNobmlxdWUiLCJhdWQiOiJtbGZsb3ciLCJzdWIiOiI0OGVlY2FiZC1kNWIzLTRlODMtYjk3MC1lNjM3M2JjNDE5YzQiLCJ0eXAiOiJJRCIsImF6cCI6Im1sZmxvdyIsInNpZCI6IjhiNDQ0MzAxLTQ2YTQtNDc1OC04Zjg3LTkzOWM0ZWNiZDViZCIsImF0X2hhc2giOiI2MFFPbDE3cWhiN0pPQ2VyRlB1MTdRIiwiYWNyIjoiMSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJuYW1lIjoicmRyIHJkciIsImdyb3VwcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy10ZWNobmlxdWUiLCJkYXRhc2NpZW50aXN0Il0sInByZWZlcnJlZF91c2VybmFtZSI6InJkciIsImdpdmVuX25hbWUiOiJyZHIiLCJmYW1pbHlfbmFtZSI6InJkciIsImVtYWlsIjoicmRyQHNpdGUuY29tIn0.t3w1qcfAQejHjvi9p26V3B6ousATwtO6MB31auCiBaVYEhiJBEfFcAqtCXhD1EFY6NSJO_38gM3XS5jR2zef2CfPbrsycRNBnER8wVEyU9qgDdNFdhKZKfqkDxkuPpq_0Avq87gMBsKlWDsw7bUHe6IXvnJ0YNdK9ctKXjhz1L6BJfLBrOOD5OMrGY0UeUgXMYtRYCh-fN8PhsSz5haepmIbD2VyXeDiAkcgtp14jVQ-jaTLOo5yuarFHhiKoCODwCc0kiiWJhPPwZX6L_Xt7MjXCKRlR0aFuWaN2XNbGOGbVy27MRY_V7-V9625LxG6CNB3O_eL2y2e0jLtziVefw"
    else:
        token = token[len(BEARER_PREFIX) :]  # Remove prefix
    try:
        token_info = jwt.decode(token, auth_config.jwt_public_key, algorithms=["HS256"], options={"verify_signature": False})
        if not token_info:  # pragma: no cover
            _logger.warning("No token_info returned")
            return error_response
        token_info["username"] = token_info[auth_config.jwt_username_key]
        _logger.info(f"User {token_info['username']} authenticated")

        # Create user if they don't exist

        if not user_exists(token_info['username']):
            create_user(token_info["username"])
            _logger.info(f"User {token_info['username']} created locally")

        return Authorization(auth_type="jwt", data=token_info)
    except Exception as e:
        _logger.error(e)

    _logger.warning("Missing or invalid authorization token")
    return error_response