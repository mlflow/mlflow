import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/redact.py"
redact = runpy.run_path(str(SCRIPT))["redact"]


@pytest.mark.parametrize(
    ("secret", "displayed"),
    [
        ("fake-token", "fake-token"),
        ('fake"token', 'fake"token'),
        ("fake/token+value", "fake%2Ftoken%2Bvalue"),
        ("fake token", "fake+token"),
        ("fake-token", "ZmFrZS10b2tlbg=="),
    ],
)
def test_redact(secret: str, displayed: str) -> None:
    message = json.dumps({"type": "assistant", "text": f"Token: {displayed}"})
    assert json.loads(redact(message, ["", secret])) == {
        "type": "assistant",
        "text": "Token: [REDACTED]",
    }


def test_redact_overlapping_secrets() -> None:
    assert redact("fake-token-extended fake-token", ["fake-token", "fake-token-extended"]) == (
        "[REDACTED] [REDACTED]"
    )


def test_redact_without_secrets() -> None:
    assert redact("Keep this text\n", [""]) == "Keep this text\n"


def test_redact_stdin_stdout() -> None:
    result = subprocess.run(
        [sys.executable, SCRIPT],
        input="first-secret\nsecond%2Fsecret\nunchanged\n",
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ | {"SECRETS_TO_REDACT": "\nfirst-secret\nsecond/secret\n\n"},
        check=True,
    )
    assert result.stdout == "[REDACTED]\n[REDACTED]\nunchanged\n"


def test_redact_missing_secrets() -> None:
    with subprocess.Popen(
        [sys.executable, SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={key: value for key, value in os.environ.items() if key != "SECRETS_TO_REDACT"},
        text=True,
    ) as process:
        assert process.wait(timeout=5) != 0
        stdout, stderr = process.communicate()
    assert stdout == ""
    assert "SECRETS_TO_REDACT must be set" in stderr
