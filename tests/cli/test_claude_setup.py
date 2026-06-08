"""Tests for the `mlflow claude-setup` CLI command."""

from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_clone(skill_names):
    """Return a side_effect for subprocess.run that populates tmpdir with dirs."""

    def _side_effect(cmd, check, capture_output):
        # cmd is: ["git", "clone", "--depth", "1", repo_url, tmpdir]
        tmpdir = cmd[-1]
        for name in skill_names:
            Path(tmpdir, name).mkdir(parents=True)
            # put a marker file so copytree has something to copy
            (Path(tmpdir, name) / "SKILL.md").write_text(f"# {name}")
        # also plant a dot-directory that should be skipped
        Path(tmpdir, ".git").mkdir()

    return _side_effect


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_claude_setup_happy_path(tmp_path):
    """Skills are cloned and copied to the target directory."""
    runner = CliRunner()
    skill_names = ["mlflow-tracking", "mlflow-evaluation"]

    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone(skill_names)) as mock_run,
        mock.patch("shutil.copytree") as mock_copytree,
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    assert mock_copytree.call_count == len(skill_names)
    for name in skill_names:
        assert f"\u2713 {name}" in result.output
    assert "Restart Claude Code" in result.output


def test_claude_setup_creates_target_dir(tmp_path):
    """Target directory is created if it does not exist."""
    runner = CliRunner()
    nested = tmp_path / "deep" / "nested" / "skills"

    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone(["skill-a"])),
        mock.patch("shutil.copytree"),
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(nested)],
        )

    assert result.exit_code == 0, result.output
    assert nested.exists()


# ---------------------------------------------------------------------------
# --force flag
# ---------------------------------------------------------------------------


def test_claude_setup_skips_existing_without_force(tmp_path):
    """Existing skills are skipped when --force is not used."""
    runner = CliRunner()
    skill_names = ["skill-a", "skill-b"]

    # Pre-create one of the skill directories to simulate it already existing
    (tmp_path / "skill-a").mkdir()

    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone(skill_names)),
        mock.patch("shutil.copytree") as mock_copytree,
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    # skill-a already existed → should be skipped, not overwritten
    assert mock_copytree.call_count == 1  # only skill-b copied
    assert "Skipped (already installed): skill-a" in result.output
    assert "Use --force to overwrite." in result.output
    assert "\u2713 skill-b" in result.output


def test_claude_setup_force_overwrites_existing(tmp_path):
    """Existing skills are removed and reinstalled when --force is used."""
    import importlib

    _module = importlib.import_module("mlflow.cli.claude_setup")

    runner = CliRunner()
    skill_names = ["skill-a", "skill-b"]

    # Pre-create both skill directories in the target
    for name in skill_names:
        (tmp_path / name).mkdir()

    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone(skill_names)),
        mock.patch.object(_module.shutil, "copytree") as mock_copytree,
        mock.patch.object(_module.shutil, "rmtree") as mock_rmtree,
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(tmp_path), "--force"],
        )

    assert result.exit_code == 0, result.output
    # Both skills should be copied
    assert mock_copytree.call_count == len(skill_names)
    # Each pre-existing skill dir should have been removed before re-copy.
    # rmtree may also be called by TemporaryDirectory cleanup, so we check
    # that each target skill path appears in the rmtree call args.
    rmtree_paths = {str(c.args[0]) for c in mock_rmtree.call_args_list}
    for name in skill_names:
        assert str(tmp_path / name) in rmtree_paths
    assert "Skipped" not in result.output


# ---------------------------------------------------------------------------
# --skills-repo option
# ---------------------------------------------------------------------------


def test_claude_setup_custom_repo(tmp_path):
    """--skills-repo is forwarded to git clone."""
    runner = CliRunner()
    custom_repo = "https://github.com/my-org/my-skills"

    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone(["skill-x"])) as mock_run,
        mock.patch("shutil.copytree"),
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(tmp_path), "--skills-repo", custom_repo],
        )

    assert result.exit_code == 0, result.output
    cmd_used = mock_run.call_args[0][0]
    assert custom_repo in cmd_used


# ---------------------------------------------------------------------------
# --target-dir option
# ---------------------------------------------------------------------------


def test_claude_setup_custom_target_dir(tmp_path):
    """--target-dir controls where skills are installed."""
    runner = CliRunner()
    custom_target = tmp_path / "custom-skills"

    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone(["skill-y"])),
        mock.patch("shutil.copytree") as mock_copytree,
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(custom_target)],
        )

    assert result.exit_code == 0, result.output
    # The destination passed to copytree should be inside custom_target
    dest_arg = mock_copytree.call_args[0][1]
    assert str(custom_target) in dest_arg


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_claude_setup_git_failure(tmp_path):
    """A failing git clone exits with a non-zero code and shows an error."""
    runner = CliRunner()

    with mock.patch(
        "subprocess.run",
        side_effect=__import__("subprocess").CalledProcessError(
            128, ["git"], stderr=b"repository not found"
        ),
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(tmp_path)],
        )

    assert result.exit_code != 0
    assert "Failed to clone" in result.output


def test_claude_setup_no_skills_in_repo(tmp_path):
    """An empty repo (no skill dirs) reports zero installed skills."""
    runner = CliRunner()

    # Clone side-effect creates no skill directories
    with (
        mock.patch("subprocess.run", side_effect=_make_fake_clone([])),
        mock.patch("shutil.copytree") as mock_copytree,
    ):
        result = runner.invoke(
            cli,
            ["claude-setup", "--target-dir", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    mock_copytree.assert_not_called()
    assert "No new skills installed" in result.output
