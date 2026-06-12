import re
from pathlib import Path

import pytest

import mlflow
from mlflow.assistant.providers import GATEWAY_PROVIDER_NAME

# The frontend hardcodes the gateway provider id in its own constant because
# there is no shared definition across the Python/TS boundary. This test fails
# if the two drift. Python (GATEWAY_PROVIDER_NAME) is the source of truth: the
# `/config` payload keys `providers` by it, so fix the TS side to match.
_CONSTANTS_TS = (
    Path(mlflow.__file__).parent / "server" / "js" / "src" / "assistant" / "constants.ts"
)
_GATEWAY_PROVIDER_ID_PATTERN = re.compile(
    r"""export\s+const\s+GATEWAY_PROVIDER_ID\s*=\s*['"]([^'"]+)['"]"""
)


def test_gateway_provider_id_matches_frontend_constant():
    if not _CONSTANTS_TS.exists():
        pytest.skip(f"frontend source not present: {_CONSTANTS_TS}")

    match = _GATEWAY_PROVIDER_ID_PATTERN.search(_CONSTANTS_TS.read_text())
    assert match is not None, (
        f"Could not find `GATEWAY_PROVIDER_ID = '...'` in {_CONSTANTS_TS}. "
        "Was the constant renamed or reformatted? Update this test or restore the export."
    )

    frontend_id = match.group(1)
    assert frontend_id == GATEWAY_PROVIDER_NAME, (
        f"Gateway provider id is out of sync: Python GATEWAY_PROVIDER_NAME="
        f"{GATEWAY_PROVIDER_NAME!r} but TS GATEWAY_PROVIDER_ID={frontend_id!r}. "
        f"Python is authoritative; update {_CONSTANTS_TS}."
    )
