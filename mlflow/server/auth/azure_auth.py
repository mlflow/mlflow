"""
Azure AD/Entra ID SSO authentication for MLflow.

This module provides OpenID Connect (OIDC) authentication integration with Azure AD.

Usage:
    1. Set environment variables:
        - AZURE_TENANT_ID: Your Azure tenant ID
        - AZURE_CLIENT_ID: Application client ID from Azure Portal
        - AZURE_CLIENT_SECRET: Client secret from Azure Portal
        - AZURE_REDIRECT_URI: OAuth callback URI (default: http://localhost:5000/login/callback)

    2. Configure MLflow auth:
        Set authorization_function in auth_config.ini to:
        mlflow.server.auth.azure_auth:authenticate_request_azure

    3. Start MLflow:
        mlflow server --app-name basic-auth
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import base64
from functools import lru_cache

import requests
from flask import Request, Response as FlaskResponse, session
from werkzeug.datastructures import Authorization

try:
    from msal import ConfidentialClientApplication
    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False

try:
    from jose import jwt, JWTClaimsValidationError
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AzureAuthConfig:
    """Configuration for Azure SSO authentication."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        authority: Optional[str] = None,
        scopes: Optional[list] = None,
        graph_api_endpoint: str = "https://graph.microsoft.com/v1.0",
    ):
        """
        Initialize Azure Auth Configuration.

        Args:
            tenant_id: Azure AD tenant ID
            client_id: OAuth application client ID
            client_secret: OAuth application client secret
            redirect_uri: Callback URI for OAuth flow
            authority: Azure authority endpoint (auto-generated if not provided)
            scopes: OAuth scopes to request (default: ["User.Read"])
            graph_api_endpoint: Microsoft Graph API endpoint
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.authority = authority or f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self.scopes = scopes or ["User.Read"]
        self.graph_api_endpoint = graph_api_endpoint

    @classmethod
    def from_env(cls) -> Optional["AzureAuthConfig"]:
        """
        Create config from environment variables.

        Returns:
            AzureAuthConfig instance if all required env vars are set, None otherwise
        """
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        redirect_uri = os.getenv("AZURE_REDIRECT_URI", "http://localhost:5000/login/callback")

        if not all([tenant_id, client_id, client_secret]):
            logger.warning(
                "Azure SSO not fully configured. "
                "Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET"
            )
            return None

        return cls(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
        )


class AzureAuthenticator:
    """Handles Azure AD authentication and token management."""

    def __init__(self, config: AzureAuthConfig):
        """
        Initialize the authenticator.

        Args:
            config: AzureAuthConfig instance
        """
        if not MSAL_AVAILABLE:
            raise ImportError("msal package is required for Azure authentication. Install with: pip install msal")

        self.config = config
        self.app = ConfidentialClientApplication(
            client_id=config.client_id,
            authority=config.authority,
            client_credential=config.client_secret,
        )
        self._token_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(hours=1)

    def get_authorization_url(self) -> str:
        """
        Generate Azure login URL.

        Returns:
            Authorization URL for user to visit
        """
        auth_url = self.app.get_authorization_request_url(
            scopes=self.config.scopes,
            redirect_uri=self.config.redirect_uri,
            state="mlflow_auth_state",
        )
        return auth_url

    def exchange_code_for_token(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth callback

        Returns:
            Token response dict with access_token, or None on failure
        """
        try:
            result = self.app.acquire_token_by_authorization_code(
                code=code,
                scopes=self.config.scopes,
                redirect_uri=self.config.redirect_uri,
            )

            if "error" in result:
                logger.error(f"Token exchange failed: {result.get('error_description')}")
                return None

            return result
        except Exception as e:
            logger.error(f"Error exchanging code for token: {str(e)}")
            return None

    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information from Microsoft Graph API.

        Args:
            access_token: Valid Azure AD access token

        Returns:
            User info dict with userPrincipalName, displayName, etc., or None on failure
        """
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(
                f"{self.config.graph_api_endpoint}/me",
                headers=headers,
                timeout=10,
            )

            if response.status_code != 200:
                logger.error(f"Failed to get user info: {response.text}")
                return None

            return response.json()
        except Exception as e:
            logger.error(f"Error fetching user info: {str(e)}")
            return None

    @lru_cache(maxsize=1)
    def _get_public_keys(self) -> Optional[list]:
        """
        Fetch public keys from Azure.

        Returns:
            List of JWK keys, or None on failure
        """
        try:
            response = requests.get(
                f"{self.config.authority}/discovery/v2.0/keys",
                timeout=10,
            )
            response.raise_for_status()
            return response.json().get("keys", [])
        except Exception as e:
            logger.error(f"Failed to fetch public keys: {str(e)}")
            return None

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded claims dict, or None if verification fails
        """
        if not JOSE_AVAILABLE:
            logger.error("python-jose package is required for token verification. Install with: pip install python-jose")
            return None

        try:
            keys = self._get_public_keys()
            if not keys:
                logger.error("Unable to fetch public keys")
                return None

            # Get token header to find correct key
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            # Find the correct key
            key = None
            for k in keys:
                if k.get("kid") == kid:
                    key = k
                    break

            if not key:
                logger.error("Unable to find key for token verification")
                return None

            # Convert JWK to PEM format
            try:
                from cryptography.hazmat.primitives.asymmetric import rsa
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.backends import default_backend
            except ImportError:
                logger.error("cryptography package required. Install with: pip install cryptography")
                return None

            n = int.from_bytes(base64.urlsafe_b64decode(key["n"] + "=="), "big")
            e = int.from_bytes(base64.urlsafe_b64decode(key["e"] + "=="), "big")

            public_key = rsa.RSAPublicNumbers(e, n).public_key(default_backend())
            pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            # Verify and decode
            claims = jwt.decode(
                token,
                pem,
                algorithms=["RS256"],
                audience=self.config.client_id,
                issuer=f"{self.config.authority}/v2.0",
                options={"verify_exp": True},
            )

            return claims
        except JWTClaimsValidationError as e:
            logger.error(f"Token validation failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None


def authenticate_request_azure(request: Request) -> Optional[Authorization]:
    """
    MLflow custom authentication function for Azure SSO.

    This function is called for every request to MLflow server.
    It checks for Azure authentication via Bearer token or OAuth code.

    Args:
        request: Flask request object

    Returns:
        Authorization object with username, None, or Flask Response with error
    """
    config = AzureAuthConfig.from_env()
    if not config:
        logger.warning("Azure configuration incomplete, falling back to no auth")
        return None

    try:
        authenticator = AzureAuthenticator(config)
    except ImportError as e:
        logger.error(f"Azure authentication not available: {str(e)}")
        response = FlaskResponse("Authentication module not configured", status=500)
        return response

    # Check for authorization header (Bearer token)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        claims = authenticator.verify_token(token)

        if claims:
            username = claims.get("preferred_username") or claims.get("unique_name")
            if username:
                logger.debug(f"Azure authentication successful for {username}")
                return Authorization("azure", {"username": username})

    # Check for authorization code (callback flow)
    code = request.args.get("code")
    if code:
        result = authenticator.exchange_code_for_token(code)
        if result:
            user_info = authenticator.get_user_info(result.get("access_token"))
            if user_info:
                username = user_info.get("userPrincipalName") or user_info.get("mail") or user_info.get("id")
                # Store token in session for future use
                try:
                    session["azure_token"] = result.get("access_token")
                except Exception as e:
                    logger.warning(f"Could not store token in session: {str(e)}")

                logger.debug(f"Azure authentication successful (via code) for {username}")
                return Authorization("azure", {"username": username})

    # No valid authentication found
    response = FlaskResponse("Unauthorized", status=401)
    response.headers["WWW-Authenticate"] = 'Bearer realm="MLflow Azure SSO"'
    return response


# Backward compatibility alias
authenticate_request_with_azure_sso = authenticate_request_azure
