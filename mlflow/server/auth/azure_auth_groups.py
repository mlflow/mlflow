"""
Azure AD/Entra ID SSO authentication for MLflow with Group-Based Access Control.

This module provides OpenID Connect (OIDC) authentication with Azure AD group validation.

Environment Variables:
    Required:
        - AZURE_TENANT_ID: Your Azure tenant ID
        - AZURE_CLIENT_ID: Application client ID from Azure Portal
        - AZURE_CLIENT_SECRET: Client secret from Azure Portal
        - AZURE_REDIRECT_URI: OAuth callback URI (default: http://localhost:5000/login/callback)
    
    Optional (Group-Based Access Control):
        - AZURE_ALLOWED_GROUPS: Comma-separated list of allowed Azure AD group IDs or names
                                Format: "group-id-1,group-id-2" or "MLflow-Users,MLflow-Admins"
        - AZURE_GROUP_CLAIM_NAME: JWT claim containing groups (default: "groups")
        - AZURE_DENY_GROUPS: Comma-separated list of groups to deny access
        - AZURE_REQUIRE_GROUP_MEMBERSHIP: "true" to enforce group membership (default: false)

Example:
    export AZURE_TENANT_ID="12345678-1234-1234-1234-123456789012"
    export AZURE_CLIENT_ID="87654321-4321-4321-4321-210987654321"
    export AZURE_CLIENT_SECRET="your-secret"
    export AZURE_REDIRECT_URI="http://localhost:5000/login/callback"
    export AZURE_ALLOWED_GROUPS="12345678-1234-1234-1234-111111111111,12345678-1234-1234-1234-222222222222"
    export AZURE_REQUIRE_GROUP_MEMBERSHIP="true"
"""

import os
import logging
from typing import Optional, Dict, Any, List, Set
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
    """Configuration for Azure SSO authentication with group validation."""

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        allowed_groups: Optional[List[str]] = None,
        deny_groups: Optional[List[str]] = None,
        group_claim_name: str = "groups",
        require_group_membership: bool = False,
        authority: Optional[str] = None,
        scopes: Optional[list] = None,
        graph_api_endpoint: str = "https://graph.microsoft.com/v1.0",
    ):
        """
        Initialize Azure Auth Configuration with group validation.

        Args:
            tenant_id: Azure AD tenant ID
            client_id: OAuth application client ID
            client_secret: OAuth application client secret
            redirect_uri: Callback URI for OAuth flow
            allowed_groups: List of allowed Azure AD group IDs/names
            deny_groups: List of denied Azure AD group IDs/names
            group_claim_name: JWT claim name containing groups
            require_group_membership: Enforce group membership check
            authority: Azure authority endpoint (auto-generated if not provided)
            scopes: OAuth scopes to request (default: ["User.Read"])
            graph_api_endpoint: Microsoft Graph API endpoint
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.allowed_groups = set(allowed_groups or [])
        self.deny_groups = set(deny_groups or [])
        self.group_claim_name = group_claim_name
        self.require_group_membership = require_group_membership
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
        
        # Group-based access control settings
        allowed_groups_str = os.getenv("AZURE_ALLOWED_GROUPS", "")
        deny_groups_str = os.getenv("AZURE_DENY_GROUPS", "")
        group_claim_name = os.getenv("AZURE_GROUP_CLAIM_NAME", "groups")
        require_group = os.getenv("AZURE_REQUIRE_GROUP_MEMBERSHIP", "false").lower() == "true"

        if not all([tenant_id, client_id, client_secret]):
            logger.warning(
                "Azure SSO not fully configured. "
                "Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET"
            )
            return None

        # Parse group lists
        allowed_groups = [g.strip() for g in allowed_groups_str.split(",") if g.strip()]
        deny_groups = [g.strip() for g in deny_groups_str.split(",") if g.strip()]

        if allowed_groups or require_group:
            logger.info(f"Group-based access control enabled. Allowed groups: {allowed_groups}")

        return cls(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            allowed_groups=allowed_groups,
            deny_groups=deny_groups,
            group_claim_name=group_claim_name,
            require_group_membership=require_group,
        )


class AzureAuthenticator:
    """Handles Azure AD authentication and token management with group validation."""

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
        """Generate Azure login URL."""
        auth_url = self.app.get_authorization_request_url(
            scopes=self.config.scopes,
            redirect_uri=self.config.redirect_uri,
            state="mlflow_auth_state",
        )
        return auth_url

    def exchange_code_for_token(self, code: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
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
        """Get user information from Microsoft Graph API."""
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
        """Fetch public keys from Azure."""
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
        """Verify and decode JWT token."""
        if not JOSE_AVAILABLE:
            logger.error("python-jose package is required for token verification. Install with: pip install python-jose")
            return None

        try:
            keys = self._get_public_keys()
            if not keys:
                logger.error("Unable to fetch public keys")
                return None

            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            key = None
            for k in keys:
                if k.get("kid") == kid:
                    key = k
                    break

            if not key:
                logger.error("Unable to find key for token verification")
                return None

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

    def validate_group_membership(self, claims: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate if user's groups are in allowed list.

        Args:
            claims: JWT claims containing group information

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get groups from JWT claims
        user_groups = claims.get(self.config.group_claim_name, [])
        if isinstance(user_groups, str):
            user_groups = [user_groups]
        
        user_groups_set = set(user_groups)
        
        logger.debug(f"User groups from token: {user_groups_set}")
        logger.debug(f"Allowed groups: {self.config.allowed_groups}")
        logger.debug(f"Denied groups: {self.config.deny_groups}")

        # Check if user is in denied groups
        if self.config.deny_groups and user_groups_set & self.config.deny_groups:
            denied = user_groups_set & self.config.deny_groups
            logger.warning(f"User denied access - member of denied groups: {denied}")
            return False, f"User is member of denied group(s): {', '.join(denied)}"

        # Check if user is in allowed groups (if configured)
        if self.config.allowed_groups:
            if not (user_groups_set & self.config.allowed_groups):
                logger.warning(f"User denied access - not in allowed groups. Has: {user_groups_set}")
                return False, f"User is not member of any allowed group. Required: {', '.join(self.config.allowed_groups)}"

        # Check if group membership is required
        if self.config.require_group_membership and not user_groups:
            logger.warning("User denied access - group membership required but none found")
            return False, "Group membership is required for access"

        logger.info(f"User group validation passed. Groups: {user_groups_set}")
        return True, None


def authenticate_request_azure_groups(request: Request) -> Optional[Authorization]:
    """
    MLflow custom authentication function for Azure SSO with group validation.

    This function validates both user credentials and Azure AD group membership.

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
            # Validate group membership
            is_valid, error_msg = authenticator.validate_group_membership(claims)
            if not is_valid:
                response = FlaskResponse(f"Access Denied: {error_msg}", status=403)
                return response

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
                # Validate group membership from token claims
                token_claims = jwt.decode(
                    result.get("id_token", ""),
                    options={"verify_signature": False}
                ) if result.get("id_token") else {}
                
                is_valid, error_msg = authenticator.validate_group_membership(token_claims)
                if not is_valid:
                    response = FlaskResponse(f"Access Denied: {error_msg}", status=403)
                    return response

                username = user_info.get("userPrincipalName") or user_info.get("mail") or user_info.get("id")
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


# Backward compatibility aliases
authenticate_request_with_azure_groups = authenticate_request_azure_groups
