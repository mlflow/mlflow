import logging
from datetime import datetime, timezone

_audit_logger = logging.getLogger("mlflow.auth.audit")


def log_login(username: str, provider: str, ip_address: str):
    _audit_logger.info(
        "LOGIN user=%s provider=%s ip=%s time=%s",
        username,
        provider,
        ip_address,
        datetime.now(timezone.utc).isoformat(),
    )


def log_logout(username: str, provider: str, ip_address: str):
    _audit_logger.info(
        "LOGOUT user=%s provider=%s ip=%s time=%s",
        username,
        provider,
        ip_address,
        datetime.now(timezone.utc).isoformat(),
    )


def log_login_failed(reason: str, provider: str, ip_address: str):
    _audit_logger.warning(
        "LOGIN_FAILED reason=%s provider=%s ip=%s time=%s",
        reason,
        provider,
        ip_address,
        datetime.now(timezone.utc).isoformat(),
    )


def log_permission_denied(username: str, resource: str, action: str, ip_address: str):
    _audit_logger.warning(
        "PERMISSION_DENIED user=%s resource=%s action=%s ip=%s time=%s",
        username,
        resource,
        action,
        ip_address,
        datetime.now(timezone.utc).isoformat(),
    )


def log_user_provisioned(username: str, provider: str, is_admin: bool):
    _audit_logger.info(
        "USER_PROVISIONED user=%s provider=%s is_admin=%s time=%s",
        username,
        provider,
        is_admin,
        datetime.now(timezone.utc).isoformat(),
    )
