"""Sample JWT authentication module for testing purposes.

NOT SUITABLE FOR PRODUCTION USE.
"""

import logging

import jwt

from mlflow.server.request_context import Authorization, get_request
from mlflow.server.responses import _CompatResponse

BEARER_PREFIX = "bearer "

_logger = logging.getLogger(__name__)


def authenticate_request() -> Authorization | _CompatResponse:
    _logger.debug("Getting token")
    error_response = _CompatResponse(
        content="You are not authenticated. Please provide a valid JWT Bearer token with the request.",
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
    )

    token = get_request().headers.get("Authorization")
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
