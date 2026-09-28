import argparse
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest
from skills.cli import build_parser
from skills.commands import upload_media
from skills.github.uploads import UploadFailed

URL = "https://github.com/user-attachments/assets/2f1c0a3e-0000-4000-8000-000000000001"
REPO_ID = "136202695"


def build_args(*argv: str) -> argparse.Namespace:
    return build_parser().parse_args(["upload-media", *argv])


@pytest.fixture
def shot(tmp_path: Path) -> Path:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")
    return path


@pytest.fixture
def credentials() -> Iterator[None]:
    # The command's own asserts pin REPO_ID and "t" through to upload_asset.
    with (
        mock.patch.object(upload_media, "get_github_token", return_value="t"),
        mock.patch.object(upload_media, "resolve_repository_id", return_value=REPO_ID),
    ):
        yield


def test_prints_a_url_for_each_file(
    tmp_path: Path, shot: Path, credentials: None, capsys: pytest.CaptureFixture[str]
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    args = build_args(str(shot), str(clip))

    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    assert uploader.call_args_list == [
        mock.call(shot, REPO_ID, "t"),
        mock.call(clip, REPO_ID, "t"),
    ]
    assert capsys.readouterr().out == f"{shot}\t{URL}\n{clip}\t{URL}\n"


def test_reports_a_failure_and_exits_nonzero(
    shot: Path, credentials: None, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_args(str(shot))

    with (
        mock.patch.object(
            upload_media, "upload_asset", side_effect=UploadFailed("shot.png: boom")
        ) as uploader,
        pytest.raises(SystemExit, match="^1$"),
    ):
        args.func(args)

    uploader.assert_called_once_with(shot, REPO_ID, "t")

    assert "failed shot.png: boom" in capsys.readouterr().err


def test_a_failed_file_does_not_stop_the_rest(
    tmp_path: Path, shot: Path, credentials: None, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    args = build_args(str(notes), str(shot))

    outcomes = [UploadFailed("notes.txt: unsupported extension"), URL]
    with (
        mock.patch.object(upload_media, "upload_asset", side_effect=outcomes) as uploader,
        pytest.raises(SystemExit, match="^1$"),
    ):
        args.func(args)

    assert uploader.call_count == 2
    assert capsys.readouterr().out == f"{shot}\t{URL}\n"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_credential_fault_stops_the_remaining_uploads(
    tmp_path: Path, shot: Path, credentials: None, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    second = tmp_path / "second.png"
    second.write_bytes(b"\x89PNG")
    args = build_args(str(shot), str(second))

    with (
        mock.patch.object(
            upload_media,
            "upload_asset",
            side_effect=UploadFailed(f"shot.png: refused ({status})", status=status),
        ) as uploader,
        pytest.raises(SystemExit, match="^1$"),
    ):
        args.func(args)

    uploader.assert_called_once_with(shot, REPO_ID, "t")
    err = capsys.readouterr().err
    assert f"failed shot.png: refused ({status})" in err
    # Only a rejected credential is worth re-authenticating for; a 403 or a 404 is not.
    assert ("check GH_TOKEN or run `gh auth login`" in err) == (status == 401)


def test_a_missing_path_is_reported_without_uploading(
    tmp_path: Path, credentials: None, capsys: pytest.CaptureFixture[str]
) -> None:
    args = build_args(str(tmp_path / "gone.png"), str(tmp_path))

    with (
        mock.patch.object(upload_media, "upload_asset") as uploader,
        pytest.raises(SystemExit, match="^1$"),
    ):
        args.func(args)

    uploader.assert_not_called()
    assert capsys.readouterr().err.count("not a file") == 2


def test_repo_defaults_to_mlflow_and_is_overridable(shot: Path) -> None:
    assert build_args(str(shot)).repo == "mlflow/mlflow"
    assert build_args(str(shot), "--repo", "harupy/mlflow").repo == "harupy/mlflow"


def test_resolve_repository_id_returns_the_id() -> None:
    completed = subprocess.CompletedProcess([], 0, stdout=f"{REPO_ID}\n", stderr="")
    with mock.patch("subprocess.run", return_value=completed) as run:
        assert upload_media.resolve_repository_id("mlflow/mlflow") == REPO_ID

    assert run.call_args.args[0] == ["gh", "api", "repos/mlflow/mlflow", "--jq", ".id"]


def test_resolve_repository_id_surfaces_the_gh_error(capsys: pytest.CaptureFixture[str]) -> None:
    failure = subprocess.CalledProcessError(1, "gh", stderr="gh: Not Found (HTTP 404)\n")
    with (
        mock.patch("subprocess.run", side_effect=failure) as run,
        pytest.raises(SystemExit, match="^1$"),
    ):
        upload_media.resolve_repository_id("mlflow/nope")

    run.assert_called_once()
    # Without this the user only sees "returned non-zero exit status 1".
    assert "gh: Not Found (HTTP 404)" in capsys.readouterr().err


@pytest.mark.parametrize(
    "outcome",
    [
        subprocess.CalledProcessError(1, "gh", stderr=""),
        FileNotFoundError("gh"),
    ],
)
def test_resolve_repository_id_exits_when_gh_fails(outcome: Exception) -> None:
    with (
        mock.patch("subprocess.run", side_effect=outcome) as run,
        pytest.raises(SystemExit, match="^1$"),
    ):
        upload_media.resolve_repository_id("mlflow/mlflow")

    run.assert_called_once()
