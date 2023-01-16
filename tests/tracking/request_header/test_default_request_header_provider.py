from mlflow.tracking.request_header.default_request_header_provider import (
    DefaultRequestHeaderProvider,
    _DEFAULT_HEADERS,
)


def test_default_request_header_provider_in_context():
    assert DefaultRequestHeaderProvider().in_context()


def test_default_request_header_provider_request_headers():
    request_headers = DefaultRequestHeaderProvider().request_headers()
    assert request_headers == _DEFAULT_HEADERS
