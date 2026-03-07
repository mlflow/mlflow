import logging

import pytest

from mlflow.server.auth.oauth.audit import (
    log_login,
    log_login_failed,
    log_logout,
    log_permission_denied,
    log_user_provisioned,
)


@pytest.fixture
def audit_records():
    logger = logging.getLogger("mlflow.auth.audit")
    records = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Collector()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    yield records
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def test_audit_logging_log_login(audit_records):
    log_login("jane", "oidc:primary", "10.0.0.1")
    assert any("LOGIN user=jane provider=oidc:primary ip=10.0.0.1" in r for r in audit_records)


def test_audit_logging_log_logout(audit_records):
    log_logout("jane", "oidc:primary", "10.0.0.1")
    assert any("LOGOUT user=jane provider=oidc:primary ip=10.0.0.1" in r for r in audit_records)


def test_audit_logging_log_login_failed(audit_records):
    log_login_failed("invalid_state", "oidc:primary", "10.0.0.1")
    assert any("LOGIN_FAILED reason=invalid_state" in r for r in audit_records)


def test_audit_logging_log_permission_denied(audit_records):
    log_permission_denied("jane", "experiment:123", "read", "10.0.0.1")
    assert any(
        "PERMISSION_DENIED user=jane resource=experiment:123 action=read" in r
        for r in audit_records
    )


def test_audit_logging_log_user_provisioned(audit_records):
    log_user_provisioned("jane", "oidc:primary", True)
    assert any(
        "USER_PROVISIONED user=jane provider=oidc:primary is_admin=True" in r for r in audit_records
    )
