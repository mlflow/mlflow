from dataclasses import dataclass, field
from unittest import mock

import pydantic
import pytest

from mlflow.tracing.utils.redaction import strip_credentials


@dataclass(frozen=True)
class _Client:
    base_url: str
    _api_key: str = field(repr=False)


@dataclass
class _Model:
    id: str
    api_key: str
    client: _Client


class _Settings(pydantic.BaseModel):
    max_tokens: int
    auth_token: str


def test_strip_credentials_drops_credential_keys_but_keeps_config():
    value = {
        "api_key": "secret",
        "Authorization": "Bearer secret",
        "accessToken": "secret",
        "client_secret": "secret",
        "max_tokens": 10,
        "tokenizer": "gpt2",
        "token_usage": {"input_tokens": 1},
        "messages": ["hi"],
    }

    assert strip_credentials(value) == {
        "max_tokens": 10,
        "tokenizer": "gpt2",
        "token_usage": {"input_tokens": 1},
        "messages": ["hi"],
    }


def test_strip_credentials_expands_nested_dataclasses_and_pydantic_models():
    value = {
        "model": _Model(id="m", api_key="secret", client=_Client("http://x", "secret")),
        "settings": _Settings(max_tokens=10, auth_token="secret"),
    }

    assert strip_credentials(value) == {
        "model": {"id": "m", "client": {"base_url": "http://x"}},
        "settings": {"max_tokens": 10},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (({"api_key": "s", "a": 1},), ({"a": 1},)),
        ([{"api_key": "s", "a": 1}], [{"a": 1}]),
        ({_Client("http://x", "s")}, [{"base_url": "http://x"}]),
    ],
)
def test_strip_credentials_handles_collections(value, expected):
    assert strip_credentials(value) == expected


def test_strip_credentials_handles_reference_cycles():
    value = {"name": "a"}
    value["self"] = value

    assert strip_credentials(value) == {"name": "a", "self": "dict"}


def test_strip_credentials_fails_closed():
    with mock.patch(
        "mlflow.tracing.utils.redaction._strip_credentials", side_effect=RuntimeError("boom")
    ) as mock_strip:
        assert strip_credentials({"a": 1}) == "<dict: omitted, sanitization failed>"
    mock_strip.assert_called_once()
