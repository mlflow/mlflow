"""Short-lived, user-scoped delegation credentials for MLflow Assistant tool subprocesses.

The Assistant runs its tools (Bash and the file tools) in a subprocess so their MLflow API calls
can run as the session owner rather than anonymously. The server mints an HMAC-signed credential
that names the user and an expiry, injects it into the subprocess, and the subprocess's MLflow
client attaches it (see
``mlflow/tracking/request_auth/assistant_delegation_request_auth_provider.py``). The server then
verifies it (see ``mlflow/server/auth/__init__.py``) and authenticates the request as that user.

The credential is bound to a single username by the signature and expires, so a leaked value
impersonates only that one user and only briefly. The signing key is generated at server startup
and shared only among the server's own worker processes: it is deliberately withheld from job
runner subprocesses and from the tool subprocesses themselves, so a credential can be minted only
by the server and never forged by code running in one of those subprocesses (which receive only a
signature). That confinement is what makes it safe to honor the credential on any route.
"""

import base64
import hashlib
import hmac
import time

from mlflow.environment_variables import _MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY

# Version prefix so the format can evolve; the signature covers it.
_VERSION = "v1"
# A minted credential only needs to outlast a single tool subprocess (bounded by the tool's own
# execution timeout, currently 120s), so keep the window a leaked value stays usable short.
_DEFAULT_TTL_SECONDS = 300


def _signing_key() -> str | None:
    # The dedicated delegation key, present only in the server's worker processes. None on a server
    # without auth (nothing to authenticate as) and in any subprocess (which cannot mint or verify).
    return _MLFLOW_ASSISTANT_DELEGATION_SIGNING_KEY.get()


def _sign(username: str, expiry_ms: int, key: str) -> str:
    message = f"{_VERSION}\n{username}\n{expiry_ms}".encode()
    digest = hmac.new(key.encode(), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


def mint_delegation_credential(
    username: str, ttl_seconds: int = _DEFAULT_TTL_SECONDS
) -> str | None:
    """Mint a signed credential attesting to ``username`` for a short window.

    Returns None when there is no signing key (a server without auth) or no username, so callers
    inject nothing and the subprocess's MLflow calls stay anonymous.
    """
    key = _signing_key()
    if not key or not username:
        return None
    expiry_ms = int(time.time() * 1000) + ttl_seconds * 1000
    return f"{_VERSION}:{username}:{expiry_ms}:{_sign(username, expiry_ms, key)}"


def verify_delegation_credential(credential: str | None) -> str | None:
    """Return the username a valid, unexpired credential attests to, else None.

    None for any malformed, mis-signed, or expired value, or when there is no signing key.
    """
    key = _signing_key()
    if not key or not credential:
        return None
    try:
        # rsplit so a username containing ':' still parses: the signature is url-safe base64 and
        # the expiry is digits, so neither of the trailing two fields contains ':'.
        head, expiry_str, sig = credential.rsplit(":", 2)
        version, username = head.split(":", 1)
        expiry_ms = int(expiry_str)
    except (ValueError, AttributeError):
        return None
    if version != _VERSION:
        return None
    if not hmac.compare_digest(sig, _sign(username, expiry_ms, key)):
        return None
    if int(time.time() * 1000) >= expiry_ms:
        return None
    return username
