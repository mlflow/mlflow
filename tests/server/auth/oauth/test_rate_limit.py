import time

import pytest

from mlflow.server.auth.oauth.rate_limit import RateLimiter


@pytest.fixture
def limiter():
    return RateLimiter(max_requests=3, window_seconds=1)


def _make_request_context(remote_addr="127.0.0.1"):
    from flask import Flask

    app = Flask(__name__)
    return app.test_request_context("/test", environ_base={"REMOTE_ADDR": remote_addr})


def test_rate_limiter_allows_requests_under_limit(limiter):
    with _make_request_context():
        assert limiter.is_rate_limited() is None
        assert limiter.is_rate_limited() is None
        assert limiter.is_rate_limited() is None


def test_rate_limiter_blocks_requests_over_limit(limiter):
    with _make_request_context():
        for _ in range(3):
            limiter.is_rate_limited()
        resp = limiter.is_rate_limited()
        assert resp is not None
        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "1"


def test_rate_limiter_separate_limits_per_ip(limiter):
    with _make_request_context("10.0.0.1"):
        for _ in range(3):
            limiter.is_rate_limited()
        assert limiter.is_rate_limited() is not None

    with _make_request_context("10.0.0.2"):
        assert limiter.is_rate_limited() is None


def test_rate_limiter_window_expires(limiter):
    with _make_request_context():
        for _ in range(3):
            limiter.is_rate_limited()
        assert limiter.is_rate_limited() is not None

    time.sleep(1.1)

    with _make_request_context():
        assert limiter.is_rate_limited() is None
