"""Sample JWT authentication module for testing purposes.

NOT SUITABLE FOR PRODUCTION USE.
"""

import logging

import jwt
from flask import Response, make_response, request
from werkzeug.datastructures import Authorization

BEARER_PREFIX = "bearer "

_logger = logging.getLogger(__name__)


def authenticate_request() -> Authorization | Response:
    _logger.debug("Getting token")
    error_response = make_response()
    error_response.status_code = 401
    error_response.set_data(
        "You are not authenticated. Please provide a valid JWT Bearer token with the request."
    )
    error_response.headers["WWW-Authenticate"] = 'Bearer error="invalid_token"'

    token = request.headers.get("Authorization")
    if token is not None and token.lower().startswith(BEARER_PREFIX):
        token = token[len(BEARER_PREFIX) :]  # Remove prefix
        try:
            # NOTE:
            # - This is a sample implementation for testing purposes only.
            # - Here we're using a hardcoded key, which is not secure.
            # - We also aren't validating that the user exists.
            token_info = jwt.decode(token, "secret", algorithms=["HS256"])
            if not token_info:  # pragma: no cover
                _logger.warning("No token_info returned")
                return error_response

            return Authorization(auth_type="jwt", data=token_info)
        except jwt.exceptions.InvalidTokenError:
            pass

    _logger.warning("Missing or invalid authorization token")
    return error_response
