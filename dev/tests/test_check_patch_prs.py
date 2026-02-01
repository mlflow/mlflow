import tempfile
from pathlib import Path
from unittest import mock

import pytest

from dev.check_patch_prs import Commit, main


@pytest.fixture
def mock_api_calls():
    with (
        mock.patch("dev.check_patch_prs.get_commits") as mock_get_commits,
        mock.patch("dev.check_patch_prs.fetch_patch_prs", return_value={1234: True}),
    ):
        mock_get_commits.side_effect = [
            [],  # First call for release branch
            [  # Second call for master branch
                Commit(sha="abc123", pr_num=1234, date="2024-01-01T00:00:00Z")
            ],
        ]
        yield


def test_cherry_pick_script_generation(mock_api_calls, capsys):
    with pytest.raises(SystemExit, match="0") as exc_info:
        main(version="2.10.1", dry_run=True)

    assert exc_info.value.code == 0
    captured = capsys.readouterr()

    assert "Cherry-pick script written to:" in captured.out
    script_path_line = [
        line for line in captured.out.split("\n") if "Cherry-pick script written to:" in line
    ][0]
    script_path = Path(script_path_line.split("Cherry-pick script written to:")[1].strip())

    try:
        assert script_path.exists()
        assert script_path == Path(tempfile.gettempdir()) / "cherry-pick.sh"

        script_content = script_path.read_text()
        assert "#!/usr/bin/env bash" in script_content
        assert "Cherry-picks for v2.10.1 -> branch-2.10" in script_content
        assert "Generated:" in script_content
        assert "If conflicts occur, resolve them and run:" in script_content
        assert "git cherry-pick --continue" in script_content
        assert "git cherry-pick abc123" in script_content

        assert script_path.stat().st_mode & 0o111
        assert "3. Run the cherry-pick script on the new branch:" in captured.out
    finally:
        if script_path.exists():
            script_path.unlink()
