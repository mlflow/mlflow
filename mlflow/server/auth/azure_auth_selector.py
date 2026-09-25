"""
Azure SSO Authentication Selector - No Config File Edit Needed

This module automatically selects the correct authentication handler based on
environment variables, without requiring edits to auth_azure.ini.

Environment Variables:
    AZURE_ENABLE_GROUP_CONTROL: "true" or "false" (default: "false")
        - If "true": Uses flexible group-based access control
        - If "false": Uses basic authentication

Usage:
    1. Set environment variables:
        export AZURE_TENANT_ID="..."
        export AZURE_CLIENT_ID="..."
        export AZURE_CLIENT_SECRET="..."
        export AZURE_ENABLE_GROUP_CONTROL="true"  (optional)
        export AZURE_ALLOWED_GROUPS="..."        (if group control enabled)

    2. In auth_azure.ini, set:
        authorization_function = mlflow.server.auth.azure_auth_selector:authenticate_request_auto

    3. The function automatically routes to the correct handler
"""

import os
import logging

logger = logging.getLogger(__name__)


def authenticate_request_auto(request):
    """
    Automatically select and route to the correct Azure authentication handler.

    Based on AZURE_ENABLE_GROUP_CONTROL environment variable:
    - "true": Uses flexible group-based access control
    - "false": Uses basic authentication (default)

    Args:
        request: Flask request object

    Returns:
        Authorization object or Flask Response
    """
    enable_group_control = os.getenv("AZURE_ENABLE_GROUP_CONTROL", "false").lower() == "true"

    if enable_group_control:
        logger.debug("Routing to flexible group-based authentication")
        from mlflow.server.auth.azure_auth_groups_flexible import (
            authenticate_request_azure_groups_flexible,
        )

        return authenticate_request_azure_groups_flexible(request)
    else:
        logger.debug("Routing to basic authentication")
        from mlflow.server.auth.azure_auth import authenticate_request_azure

        return authenticate_request_azure(request)
