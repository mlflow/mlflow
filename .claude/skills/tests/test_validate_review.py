import json
from pathlib import Path
from typing import Any

import pytest
from skills.cli import build_parser
from skills.commands.validate_review import format_path

BODY = "Overall the change looks good."
CRITICAL = {"path": "a.py", "body": "🔴 **CRITICAL:** unhandled None", "line": 1, "side": "RIGHT"}


def run_cli(tmp_path: Path, payload: dict[str, Any]) -> None:
    path = tmp_path / "review-payload.json"
    path.write_text(json.dumps(payload))
    args = build_parser().parse_args(["validate-review", str(path)])
    args.func(args)


def test_cli_reports_the_event_and_comment_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"event": "COMMENT", "body": BODY, "comments": [CRITICAL]}
    run_cli(tmp_path, payload)
    assert capsys.readouterr().out.strip() == "OK: event=COMMENT, comments=1"


def test_cli_exits_nonzero_and_reports_each_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "event": "APPROVE",
        "body": "",
        "comments": [{"path": "a.py", "body": "missing the prefix", "line": 0, "side": "MIDDLE"}],
    }
    with pytest.raises(SystemExit, match="^1$"):
        run_cli(tmp_path, payload)

    err = capsys.readouterr().err
    assert "failed schema validation" in err
    for path in ("  body:", "  comments[0].body:", "  comments[0].line:", "  comments[0].side:"):
        assert path in err
    assert err.index("  body:") < err.index("  comments[0].body:")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ([], "<root>"),
        (["event"], "event"),
        (["comments", 0, "body"], "comments[0].body"),
        ([0], "[0]"),
    ],
)
def test_format_path(path: list[str | int], expected: str) -> None:
    assert format_path(path) == expected
