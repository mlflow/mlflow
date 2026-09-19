import subprocess
from unittest import mock

import pytest
from skills.github import utils


def test_resolve_prefers_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "from-env")

    with mock.patch("skills.github.utils.subprocess.check_output") as gh:
        assert utils.resolve_github_token() == "from-env"

    gh.assert_not_called()


def test_resolve_falls_back_to_the_gh_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)

    with mock.patch("skills.github.utils.subprocess.check_output", return_value="from-gh\n") as gh:
        assert utils.resolve_github_token() == "from-gh"

    gh.assert_called_once_with(["gh", "auth", "token"], text=True)


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(), subprocess.CalledProcessError(1, "gh")],
    ids=["gh-not-installed", "gh-not-logged-in"],
)
def test_resolve_returns_none_when_the_gh_cli_cannot_answer(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)

    with mock.patch("skills.github.utils.subprocess.check_output", side_effect=error) as gh:
        assert utils.resolve_github_token() is None

    gh.assert_called_once()


def test_get_github_token_exits_when_nothing_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        mock.patch.object(utils, "resolve_github_token", return_value=None) as resolver,
        pytest.raises(SystemExit, match="^1$"),
    ):
        utils.get_github_token()

    resolver.assert_called_once()
    assert "GH_TOKEN not found" in capsys.readouterr().err
