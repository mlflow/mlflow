# mlflow/utils/auth.py
import os
import logging
from typing import Callable, Dict, Optional

_AUTH_TOKEN_FILE = os.environ.get("MLFLOW_AUTH_TOKEN_FILE")
_global_auth_provider: Optional[Callable[[], Dict[str, str]]] = None

def set_auth_provider(provider: Callable[[], Dict[str, str]]) -> None:
    """Set a process-global auth provider callable that returns headers per request."""
    global _global_auth_provider
    _global_auth_provider = provider

def clear_auth_provider() -> None:
    global _global_auth_provider
    _global_auth_provider = None

def _read_token_file() -> Optional[str]:
    if not _AUTH_TOKEN_FILE:
        return None
    try:
        with open(_AUTH_TOKEN_FILE, "r") as f:
            return f.read().strip()
    except Exception:
        logging.debug("Could not read MLFLOW_AUTH_TOKEN_FILE", exc_info=True)
        return None

def get_auth_headers() -> Dict[str, str]:
    """Return auth headers from provider or token file. Non-throwing; returns {} on failure."""
    try:
        if _global_auth_provider:
            try:
                hdrs = _global_auth_provider()
                return hdrs or {}
            except Exception:
                logging.exception("auth_provider callable raised; falling back")
                # fallthrough to token-file if available

        token = _read_token_file()
        if token:
            return {"Authorization": f"Bearer {token}"}
    except Exception:
        logging.exception("Unexpected error in get_auth_headers")
    return {}
