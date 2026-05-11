import pytest

from mlflow.rfunc.backend import _r_quote


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("simple", "simple"),
        ("with'quote", "with\\'quote"),
        ("with\\back", "with\\\\back"),
        ("'); system('id'); c('", "\\'); system(\\'id\\'); c(\\'"),
    ],
)
def test_r_quote_escapes_r_string_literal(value, expected):
    assert _r_quote(value) == expected
