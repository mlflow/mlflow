import tempfile
from pathlib import Path
from unittest.mock import patch

from dev.check_patch_prs import Commit


def test_generated_script_contains_branch_guard():
    version = "2.10.1"
    release_branch = "branch-2.10"

    with (
        patch("dev.check_patch_prs.get_commits") as mock_get_commits,
        patch("dev.check_patch_prs.fetch_patch_prs") as mock_fetch_patch_prs,
    ):
        # Setup mocks
        mock_get_commits.side_effect = [
            # First call for release branch
            [],
            # Second call for master
            [
                Commit(sha="abc123", pr_num=1, date="2024-01-01T00:00:00Z"),
                Commit(sha="def456", pr_num=2, date="2024-01-02T00:00:00Z"),
                Commit(sha="ghi789", pr_num=3, date="2024-01-03T00:00:00Z"),
            ],
        ]
        mock_fetch_patch_prs.return_value = {
            1: True,
            2: True,
            3: True,
        }

        # Import main here to use the patched functions
        from dev.check_patch_prs import main

        # Run the main function
        try:
            main(version, dry_run=True)
        except SystemExit:
            pass  # Expected to exit

        # Read the generated script
        tmp_dir = Path(tempfile.gettempdir())
        script_path = tmp_dir / "cherry-pick.sh"

        assert script_path.exists(), "Script file should be generated"

        script_content = script_path.read_text()

        # Verify the script contains all expected elements
        assert "set -euo pipefail" in script_content, "Script should use set -euo pipefail"
        assert "current_branch=$(git rev-parse --abbrev-ref HEAD)" in script_content, (
            "Script should get current branch"
        )
        assert 'if [[ "$current_branch" == "master" ]]' in script_content, (
            "Script should check for master branch"
        )
        assert "ERROR: This script must not be run on the master branch." in script_content, (
            "Script should have error message"
        )
        assert f"Please checkout a release branch (e.g., {release_branch})" in script_content, (
            "Script should suggest correct release branch"
        )
        assert "git cherry-pick abc123 def456 ghi789" in script_content, (
            "Script should contain cherry-pick command"
        )

        # Verify the guard comes before the cherry-pick command
        # Split by lines to find the actual command (not in comments)
        lines = script_content.split("\n")
        set_pipefail_line = next(i for i, line in enumerate(lines) if "set -euo pipefail" in line)
        guard_line = next(i for i, line in enumerate(lines) if "current_branch=" in line)
        cherry_pick_line = next(
            i
            for i, line in enumerate(lines)
            if "git cherry-pick" in line and not line.strip().startswith("#")
        )
        assert set_pipefail_line < guard_line < cherry_pick_line, (
            "Script structure should be: set -euo pipefail, then guard, then cherry-pick"
        )


def test_generated_script_is_valid_bash():
    version = "2.10.1"

    with (
        patch("dev.check_patch_prs.get_commits") as mock_get_commits,
        patch("dev.check_patch_prs.fetch_patch_prs") as mock_fetch_patch_prs,
    ):
        # Setup mocks
        mock_get_commits.side_effect = [
            [],  # Release branch
            [Commit(sha="abc123", pr_num=1, date="2024-01-01T00:00:00Z")],  # Master
        ]
        mock_fetch_patch_prs.return_value = {1: True}

        from dev.check_patch_prs import main

        try:
            main(version, dry_run=True)
        except SystemExit:
            pass

        # Read the generated script
        tmp_dir = Path(tempfile.gettempdir())
        script_path = tmp_dir / "cherry-pick.sh"

        # Verify the script is executable
        assert script_path.exists()
        assert script_path.stat().st_mode & 0o111, "Script should be executable"
