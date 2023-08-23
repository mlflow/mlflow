import os

import pytest

from mlflow.environment_variables import _BooleanEnvironmentVariable, _EnvironmentVariable


@pytest.mark.parametrize("value", [0, 1, "0", "1", "TRUE", "FALSE"])
def test_boolean_environment_variable_invalid_default_value(value):
    with pytest.raises(ValueError, match=r"must be one of \[True, False, None\]"):
        _BooleanEnvironmentVariable("TEST_BOOLEAN_ENV_VAR", value)


@pytest.mark.parametrize("value", [True, False, None])
def test_boolean_environment_variable_valid_default_value(value):
    assert _BooleanEnvironmentVariable("TEST_BOOLEAN_ENV_VAR", value).get() is value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("TRUE", True),
        ("true", True),
        ("tRuE", True),
        ("FALSE", False),
        ("false", False),
        ("FaLsE", False),
        ("1", True),
        ("0", False),
    ],
)
def test_boolean_environment_variable_valid_value(monkeypatch, value, expected):
    monkeypatch.setenv("TEST_BOOLEAN_ENV_VAR", value)
    assert _BooleanEnvironmentVariable("TEST_BOOLEAN_ENV_VAR", None).get() is expected


@pytest.mark.parametrize(
    "value",
    ["bool", "10", "yes", "no"],
)
def test_boolean_environment_variable_invalid_value(monkeypatch, value):
    monkeypatch.setenv("TEST_BOOLEAN_ENV_VAR", value)
    with pytest.raises(ValueError, match=r"must be one of \['true', 'false', '1', '0'\]"):
        _BooleanEnvironmentVariable("TEST_BOOLEAN_ENV_VAR", None).get()


def test_empty_value(monkeypatch):
    monkeypatch.setenv("TEST_ENV_VAR", "")
    assert _EnvironmentVariable("TEST_ENV_VAR", str, None).get() == ""


@pytest.mark.parametrize(
    ("var_name", "var_value", "var_type", "default_value", "expected_raw", "expected_get"),
    [
        ("TEST_VARIABLE", "123", str, "default", "123", "123"),
        ("NON_EXISTENT_VARIABLE", None, str, "default", None, "default"),
        ("TEST_VARIABLE", "", str, "default", "", ""),
        ("TEST_VARIABLE", " ", str, "default", " ", " "),
        ("TEST_VARIABLE", "123", int, 456, "123", 123),
        ("TEST_VARIABLE", "123.456", float, 789.0, "123.456", 123.456),
        ("TEST_VARIABLE", "123456789123456789", int, 123, "123456789123456789", 123456789123456789),
    ],
)
def test_environment_variable_functionality(
    monkeypatch, var_name, var_value, var_type, default_value, expected_raw, expected_get
):
    if var_value is not None:
        monkeypatch.setenv(var_name, var_value)
    env_var = _EnvironmentVariable(var_name, var_type, default_value)

    # Test if variable is defined
    assert env_var.defined == (var_value is not None)

    # Test getting raw value
    assert env_var.get_raw() == expected_raw

    # Test getting value
    assert env_var.get() == expected_get

    # Test setting and unsetting value
    env_var.set(str(default_value))
    assert os.getenv(var_name) == str(default_value)
    env_var.unset()
    assert os.getenv(var_name) is None


def test_format():
    env_var = _EnvironmentVariable("foo", str, "")
    assert f"{env_var} bar" == "foo bar"
    assert f"{env_var!r} bar" == "'foo' bar"
