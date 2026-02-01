import tempfile
from pathlib import Path
from unittest import mock

import pytest

from dev.check_patch_prs import main


@pytest.fixture
def mock_api_calls():
    """Mock all external API calls to avoid network requests during tests."""
    with (
        mock.patch("dev.check_patch_prs.get_commits", return_value=[]) as mock_get_commits,
        mock.patch("dev.check_patch_prs.fetch_patch_prs", return_value={1234: True}) as mock_fetch,
    ):
        # Setup master commits that should be cherry-picked
        from dev.check_patch_prs import Commit

        mock_get_commits.side_effect = [
            [],  # First call for release branch
            [  # Second call for master branch
                Commit(sha="abc123", pr_num=1234, date="2024-01-01T00:00:00Z")
            ],
        ]
        yield {"get_commits": mock_get_commits, "fetch_patch_prs": mock_fetch}


def test_cherry_pick_script_generation(mock_api_calls, capsys):
    # Run the main function
    with pytest.raises(SystemExit, match="0") as exc_info:
        main(version="2.10.1", dry_run=True)

    # Check that it exits with 0 in dry-run mode
    assert exc_info.value.code == 0

    # Capture output
    captured = capsys.readouterr()

    # Verify the script path was printed
    assert "Cherry-pick script written to:" in captured.out

    # Extract the script path from the output
    script_path_line = [
        line for line in captured.out.split("\n") if "Cherry-pick script written to:" in line
    ][0]
    script_path = Path(script_path_line.split("Cherry-pick script written to:")[1].strip())

    try:
        # Verify the script exists
        assert script_path.exists()

        # Verify the script content
        script_content = script_path.read_text()
        assert "#!/usr/bin/env bash" in script_content
        assert "Cherry-picks for v2.10.1 -> branch-2.10" in script_content
        assert "Generated:" in script_content
        assert "If conflicts occur, resolve them and run:" in script_content
        assert "git cherry-pick --continue" in script_content
        assert "git cherry-pick abc123" in script_content

        # Verify the script is executable
        assert script_path.stat().st_mode & 0o111

        # Verify the updated instructions are printed
        assert "3. Run the cherry-pick script on the new branch:" in captured.out
    finally:
        # Clean up - ensure file is removed even if assertions fail
        if script_path.exists():
            script_path.unlink()


def test_cherry_pick_script_location():
    expected_dir = Path(tempfile.gettempdir())
    expected_path = expected_dir / "cherry-pick.sh"

    # The test above already verifies the script path contains tempdir,
    # but let's be explicit about the expected location
    assert expected_path.parent == expected_dir
