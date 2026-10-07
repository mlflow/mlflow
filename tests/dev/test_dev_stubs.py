import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from mlflow.assistant.providers.claude_code import ClaudeCodeProvider
from mlflow.assistant.types import EventType

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_STUBS = REPO_ROOT / "dev" / "dev_stubs"
CLAUDE_CLI = DEV_STUBS / "claude_cli.py"


def _load(name: str, path: Path):
    # dev/ is not an installed package; load this module directly by path. It only
    # imports stdlib and references siblings as files, so standalone load works.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass can resolve the module's globals by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dev_stubs = _load("mlflow_dev_stubs", DEV_STUBS / "__init__.py")


# --- claude CLI stub ---------------------------------------------------------


def run_claude(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, CLAUDE_CLI, *args],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_claude_auth_probe_exits_zero_with_valid_json():
    result = run_claude("-p", "hi", "--max-turns", "1", "--output-format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["type"] == "result"
    assert payload["is_error"] is False


def test_claude_stream_json_parses_into_message_and_done_events(tmp_path):
    # Mirror the real provider's invocation: message via stdin (--input-format
    # text) and the system prompt via --append-system-prompt-file.
    system_prompt_file = tmp_path / "system_prompt.txt"
    system_prompt_file.write_text("ignored prompt with --output-format text and --resume decoys")
    result = run_claude(
        "-p",
        "--input-format",
        "text",
        "--output-format",
        "stream-json",
        "--verbose",
        "--append-system-prompt-file",
        str(system_prompt_file),
        stdin="hello",
    )
    assert result.returncode == 0

    provider = ClaudeCodeProvider()
    events = [
        event
        for line in result.stdout.splitlines()
        if line.strip()
        if (event := provider._parse_message_to_event(json.loads(line))) is not None
    ]
    assert [e.type for e in events] == [EventType.MESSAGE, EventType.DONE]
    assert events[0].data["message"]["content"][0]["text"]
    assert events[1].data["session_id"]


def test_claude_resume_reuses_session_id():
    result = run_claude("-p", "hi", "--output-format", "stream-json", "--resume", "sess-abc")
    assert result.returncode == 0
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert {e["session_id"] for e in events if "session_id" in e} == {"sess-abc"}


# --- stub registry -----------------------------------------------------------


def test_install_stubs_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown stub"):
        dev_stubs.install_stubs(["not-a-stub"])


def test_install_stubs_cleans_up_temp_dirs_on_partial_failure(monkeypatch):
    tmp_root = Path(tempfile.gettempdir())
    before = set(tmp_root.glob("mlflow-dev-stub-bin-*"))

    def boom(_result):
        raise RuntimeError("boom")

    monkeypatch.setitem(dev_stubs._INSTALLERS, "boom", boom)
    with pytest.raises(RuntimeError, match="boom"):
        # claude stages a temp dir, then `boom` fails -- the dir must not leak.
        dev_stubs.install_stubs(["claude", "boom"])

    assert set(tmp_root.glob("mlflow-dev-stub-bin-*")) == before


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the claude shim is a POSIX shell script; run_dev_server (its consumer) is POSIX-only",
)
def test_install_claude_stages_working_shim():
    result = dev_stubs.install_stubs(["claude"])
    try:
        (shim_dir,) = result.path_prepend
        probe = subprocess.run(
            [shim_dir / "claude", "-p", "hi", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert probe.returncode == 0
        assert json.loads(probe.stdout)["type"] == "result"
    finally:
        for path in result.cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)
