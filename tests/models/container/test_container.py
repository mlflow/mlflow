import os
import string

import pytest

from mlflow.models.container import (
    _derive_primary_port,
    _derive_upstream_port,
    _interpolate_nginx_config,
    _parse_sagemaker_safe_port_range,
    _select_port_from_range,
    SAGEMAKER_BIND_TO_PORT,
    SAGEMAKER_SAFE_PORT_RANGE,
)


@pytest.fixture
def default_port():
    return "8080"


@pytest.fixture
def sagemaker_bind_to_port_env_value():
    return "8081"


@pytest.fixture
def sagemaker_safe_port_range_env_value():
    return "1000-1005"


@pytest.fixture
def sagemaker_safe_port_range_env_value_parsed():
    return 1000, 1005


@pytest.fixture
def mock_sagemaker_bind_to_port_env(monkeypatch, sagemaker_bind_to_port_env_value):
    monkeypatch.setenv(SAGEMAKER_BIND_TO_PORT, sagemaker_bind_to_port_env_value)


@pytest.fixture
def mock_sagemaker_bind_to_port_env_missing(monkeypatch):
    monkeypatch.delenv(SAGEMAKER_BIND_TO_PORT, raising=False)


@pytest.fixture
def mock_sagemaker_safe_port_range_env(monkeypatch, sagemaker_safe_port_range_env_value):
    monkeypatch.setenv(SAGEMAKER_SAFE_PORT_RANGE, sagemaker_safe_port_range_env_value)


@pytest.fixture
def mock_sagemaker_safe_port_range_env_missing(monkeypatch):
    monkeypatch.delenv(SAGEMAKER_SAFE_PORT_RANGE, raising=False)


@pytest.fixture
def dummy_nginx_config_template():
    return "${primary_port} ${upstream_host} ${upstream_port} $http_host"


@pytest.fixture
def dummy_nginx_config(dummy_nginx_config_template):
    def _dummy_nginx_config(primary_port, upstream_host, upstream_port):
        return string.Template(dummy_nginx_config_template).safe_substitute(
            upstream_host=upstream_host,
            primary_port=primary_port,
            upstream_port=upstream_port,
        )

    return _dummy_nginx_config


@pytest.fixture
def dummy_nginx_config_template_file(tmp_path, dummy_nginx_config_template):
    nginx_config_path = tmp_path / "nginx.conf"
    with nginx_config_path.open("w") as file:
        file.write(dummy_nginx_config_template)
    return nginx_config_path


def test_derive_primary_port(
    mock_sagemaker_bind_to_port_env, sagemaker_bind_to_port_env_value, default_port
):
    default_port = int(default_port)
    port = _derive_primary_port(default_port)
    assert port == int(sagemaker_bind_to_port_env_value)


def test_derive_primary_port_env_missing(
    mock_sagemaker_bind_to_port_env_missing,
    sagemaker_bind_to_port_env_value,
    default_port,
):
    default_port = int(default_port)
    port = _derive_primary_port(default_port)
    assert port == default_port


def test_parse_sagemaker_safe_port_range_valid_value(
    sagemaker_safe_port_range_env_value,
    sagemaker_safe_port_range_env_value_parsed,
):
    assert (
        _parse_sagemaker_safe_port_range(sagemaker_safe_port_range_env_value)
        == sagemaker_safe_port_range_env_value_parsed
    )


@pytest.mark.parametrize(
    ("port_range", "error_message"),
    (
        ("", r"not enough values to unpack \(expected 2, got 1\)"),
        ("1000", r"not enough values to unpack \(expected 2, got 1\)"),
        ("1000-", r"invalid literal for int\(\) with base 10.*"),
        ("mlflow", r"not enough values to unpack \(expected 2, got 1\)"),
        ("ml-flow", r"invalid literal for int\(\) with base 10.*"),
    ),
)
def test_parse_sagemaker_safe_port_range_invalid_values(port_range, error_message):
    with pytest.raises(ValueError, match=error_message):
        _ = _parse_sagemaker_safe_port_range(port_range)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "busy_port", "expected"),
    (
        (1000, 1005, 8080, 1000),
        (1000, 1005, 1000, 1001),
    ),
)
def test_select_port_from_range_valid_values(lower_bound, upper_bound, busy_port, expected):
    assert _select_port_from_range(lower_bound, upper_bound, busy_port) == expected


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "busy_port", "error_message"),
    (
        (1005, 1000, 8080, r"The lower bound port value must be less than or equal.*"),
        (1000, 1000, 1000, r"Could not find a vacant port within an inclusive range.*"),
    ),
)
def test_select_port_from_range_invalid_values(lower_bound, upper_bound, busy_port, error_message):
    with pytest.raises(ValueError, match=error_message):
        _ = _select_port_from_range(lower_bound, upper_bound, busy_port)


def test_derive_upstream_port_env(
    mock_sagemaker_safe_port_range_env,
    default_port,
    sagemaker_safe_port_range_env_value_parsed,
):
    default_port = int(default_port)
    busy_port = default_port + 1
    expected_port = sagemaker_safe_port_range_env_value_parsed[0]
    assert _derive_upstream_port(busy_port, default_port) == expected_port


def test_derive_upstream_port_env_missing(mock_sagemaker_safe_port_range_env_missing, default_port):
    default_port = int(default_port)
    busy_port = default_port + 1
    assert _derive_upstream_port(busy_port, default_port) == default_port


def test_derive_upstream_port_env_missing_invalid_input(
    mock_sagemaker_safe_port_range_env_missing,
    default_port,
):
    default_port = int(default_port)
    with pytest.raises(ValueError, match=r"Default upstream port is already busy\."):
        _ = _derive_upstream_port(default_port, default_port)


def test_interpolate_nginx_config(dummy_nginx_config_template_file, dummy_nginx_config):
    primary_port = 8080
    upstream_host = "0.0.0.0"
    upstream_port = 8000
    expected = dummy_nginx_config(primary_port, upstream_host, upstream_port)

    nginx_config_file_path = _interpolate_nginx_config(
        dummy_nginx_config_template_file,
        upstream_host,
        primary_port,
        upstream_port,
    )
    nginx_conf_content = _read_file(nginx_config_file_path)

    assert nginx_conf_content == expected

    os.unlink(nginx_config_file_path)


def _read_file(path: str) -> str:
    with open(path) as file:
        return file.read()
