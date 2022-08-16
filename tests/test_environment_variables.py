from mlflow.environment_variables import _BooleanEnvironmentVariable
import pytest


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
