import argparse
import io
import sys
from pathlib import Path
from unittest import mock

import pytest
from skills.cli import build_parser
from skills.commands import embed_media
from skills.commands.embed_media import substitute
from skills.github import uploads

URL = "https://github.com/user-attachments/assets/2f1c0a3e-0000-4000-8000-000000000001"
MEDIA = "/tmp/review-media"
SHOT = f"{MEDIA}/shot.png"
CLIP = f"{MEDIA}/clip.mp4"
URLS = {SHOT: URL}


def build_args(media: Path) -> argparse.Namespace:
    return build_parser().parse_args([
        "embed-media",
        "--dir",
        str(media),
        "--repository-id",
        "136202695",
    ])


def build_check_args(media: Path) -> argparse.Namespace:
    return build_parser().parse_args([
        "embed-media",
        "--dir",
        str(media),
        "--check",
    ])


def make_media(tmp_path: Path, name: str = "shot.png") -> Path:
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    (media / name).write_bytes(b"\x89PNG")
    return media


def check(tmp_path: Path, body: str, name: str = "shot.png") -> embed_media.CheckReport:
    media = make_media(tmp_path, name)
    return embed_media.check_media(media, [body.format(p=media / name, media=media)])


def run_cli(
    args: argparse.Namespace,
    text: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[str, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    args.func(args)
    captured = capsys.readouterr()
    return captured.out, captured.err


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"![alt]({SHOT})", f"![alt]({URL})"),
        (f"[link]({SHOT})", f"[link]({URL})"),
        ("nothing to do", "nothing to do"),
        ("shot.png stays", "shot.png stays"),
        (f"`{SHOT}` stays", f"`{SHOT}` stays"),
    ],
)
def test_substitute_rewrites_each_reference_form(text: str, expected: str) -> None:
    assert substitute(text, URLS) == expected


def test_substitute_is_idempotent() -> None:
    once = substitute(f"see ![alt]({SHOT})", URLS)
    assert substitute(once, URLS) == once


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"![repro]({CLIP})", f"\n{URL}\n"),
        (f"![repro]({CLIP}) \t", f"\n{URL}\n \t"),
        (f"before\n![repro]({CLIP})\nafter", f"before\n\n{URL}\n\nafter"),
        # ![]() around a video URL renders as a broken image, so it must become a link.
        (f"see ![repro]({CLIP}) inline", f"see [repro]({URL}) inline"),
    ],
)
def test_substitute_promotes_a_standalone_video_to_a_bare_url(text: str, expected: str) -> None:
    assert substitute(text, {CLIP: URL}) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"![a screenshot]({SHOT})", "a screenshot"),
        (f"[a screenshot]({SHOT})", "a screenshot"),
        (f"![]({SHOT})", "shot.png"),
    ],
)
def test_substitute_strips_markup_for_media_that_never_uploaded(text: str, expected: str) -> None:
    assert substitute(text, {}, [SHOT]) == expected


def test_cli_converts_stdin_and_keeps_stdout_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    (media / "scratch.png").write_bytes(b"\x89PNG")
    text = f"evidence: ![the bug]({media / 'shot.png'})\n \t"
    monkeypatch.setenv("GH_TOKEN", "t")
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        out, err = run_cli(build_args(media), text, monkeypatch, capsys)
    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert out == f"evidence: ![the bug]({URL})\n \t"
    assert "Uploading 1 referenced file" in err
    assert "not referenced, skipping: scratch.png" in err
    assert "Embedded 1 of 1" in err


@pytest.mark.parametrize("text", ["", "a prose-only finding\n \t", "not json {"])
def test_cli_passes_through_text_without_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    media = make_media(tmp_path)
    with (
        mock.patch.object(embed_media, "resolve_github_token") as resolver,
        mock.patch.object(embed_media, "upload_asset") as uploader,
    ):
        out, _ = run_cli(build_args(media), text, monkeypatch, capsys)
    resolver.assert_not_called()
    uploader.assert_not_called()
    assert out == text


@pytest.mark.parametrize("missing_token", [False, True])
def test_cli_strips_local_path_when_upload_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    missing_token: bool,
) -> None:
    media = make_media(tmp_path)
    with (
        mock.patch.object(
            embed_media, "resolve_github_token", return_value=None if missing_token else "t"
        ) as resolver,
        mock.patch.object(
            embed_media, "upload_asset", side_effect=uploads.UploadFailed("shot.png: boom")
        ) as uploader,
    ):
        out, err = run_cli(
            build_args(media), f"evidence: ![the bug]({media / 'shot.png'})", monkeypatch, capsys
        )
    resolver.assert_called_once()
    if missing_token:
        uploader.assert_not_called()
        assert "no GitHub token" in err
    else:
        uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
        assert "failed shot.png: boom" in err
    assert out == "evidence: the bug"


def test_cli_never_reads_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    secret = tmp_path / "environ"
    secret.write_text("GH_TOKEN=supersecret")
    (media / "shot.png").symlink_to(secret)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        out, err = run_cli(
            build_args(media), f"![the secret]({media / 'shot.png'})", monkeypatch, capsys
        )
    uploader.assert_not_called()
    assert out == "the secret"
    assert "skip shot.png: symlink" in err


def test_cli_stops_after_a_fatal_upload_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    (media / "second.png").write_bytes(b"\x89PNG")
    monkeypatch.setenv("GH_TOKEN", "t")
    with mock.patch.object(
        embed_media,
        "upload_asset",
        side_effect=uploads.UploadFailed("the credential was rejected (401)", status=401),
    ) as uploader:
        out, err = run_cli(
            build_args(media),
            f"![the bug]({media / 'shot.png'}) and ![more]({media / 'second.png'})",
            monkeypatch,
            capsys,
        )
    assert uploader.call_count == 1
    assert out == "the bug and more"
    assert err.count("::warning::media upload stopped: the credential was rejected (401)") == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"![x]({SHOT})", True),
        (f"[x]({SHOT})", True),
        # A path that only ends with the cited one is not a reference.
        (f"![x]({MEDIA}/longshot.png)", False),
        ("![x](shot.png)", False),
        (f"`{SHOT}`", False),
        ("nothing here", False),
    ],
)
def test_is_referenced_matches_only_a_link_to_the_full_path(text: str, expected: bool) -> None:
    assert embed_media.is_referenced(SHOT, text) is expected


def test_cli_does_not_upload_a_name_that_is_a_suffix_of_a_cited_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path, "longshot.png")
    (media / "shot.png").write_bytes(b"\x89PNG")

    monkeypatch.setenv("GH_TOKEN", "t")
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        out, _ = run_cli(build_args(media), f"![x]({media / 'longshot.png'})", monkeypatch, capsys)

    uploader.assert_called_once_with(media / "longshot.png", "136202695", "t")
    assert out == f"![x]({URL})"


@pytest.mark.parametrize("body", ["![the bug]({p})", "[the bug]({p})"])
def test_check_accepts_every_form_the_rewriter_understands(tmp_path: Path, body: str) -> None:
    report = check(tmp_path, body)
    assert report.errors == []
    assert report.warnings == []
    assert report.cited == ["shot.png"]


def test_check_rejects_a_citation_naming_no_captured_file(tmp_path: Path) -> None:
    report = check(tmp_path, "![the bug]({media}/shto.png)")
    assert len(report.errors) == 1
    assert "no such file" in report.errors[0]


@pytest.mark.parametrize("body", ["![the bug](shot.png)", "![the bug](./shot.png)"])
def test_check_rejects_a_citation_that_is_not_the_path_written(tmp_path: Path, body: str) -> None:
    report = check(tmp_path, body)
    assert len(report.errors) == 1
    assert report.errors[0].endswith(f"cite the capture as {tmp_path / 'media' / 'shot.png'}")
    # The capture is plainly meant to be shown, so it is not also reported as uncited.
    assert report.warnings == []


@pytest.mark.parametrize(
    "body",
    [
        "[the docs icon](docs/static/img/logo.png)",
        "[upstream](https://example.com/diagram.png)",
        "the `logo.png` in the diff",
        "[the module](mlflow/utils.py)",
    ],
)
def test_check_leaves_references_that_are_not_captures_alone(tmp_path: Path, body: str) -> None:
    media = tmp_path / "media"
    media.mkdir()
    assert embed_media.check_media(media, [body]).errors == []


@pytest.mark.parametrize(
    "body",
    [
        "[the docs icon](docs/static/img/shot.png)",
        "[upstream](https://example.com/shot.png)",
        "[a capture from another run](/tmp/other/shot.png)",
    ],
)
def test_check_leaves_a_link_whose_basename_collides_with_a_capture_alone(
    tmp_path: Path, body: str
) -> None:
    # The capture is named shot.png, so every one of these shares its basename.
    report = check(tmp_path, body)
    assert report.errors == []


def test_check_warns_about_a_capture_nothing_cites(tmp_path: Path) -> None:
    report = check(tmp_path, "a prose-only finding")
    assert report.errors == []
    assert report.warnings == ["shot.png: cited by nothing, so it is not uploaded"]


def test_check_rejects_a_cited_file_with_an_unsupported_extension(tmp_path: Path) -> None:
    report = check(tmp_path, "[notes]({p})", "notes.txt")
    assert report.errors == ["notes.txt: unsupported extension, so the reference is dropped"]


def test_check_rejects_a_cited_file_that_is_empty(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "shot.png").write_bytes(b"")
    report = embed_media.check_media(media, [f"![the bug]({media / 'shot.png'})"])
    assert report.errors == ["shot.png: empty, so the reference is dropped"]


def test_check_rejects_a_cited_file_over_the_size_cap(tmp_path: Path) -> None:
    with mock.patch.object(uploads, "MAX_IMAGE_BYTES", 2):
        report = check(tmp_path, "![the bug]({p})")
    assert len(report.errors) == 1
    assert "exceeds the 2 byte cap" in report.errors[0]


def test_check_warns_when_a_video_is_cited_mid_paragraph(tmp_path: Path) -> None:
    report = check(tmp_path, "the repro is [here]({p}) inline", "clip.mp4")
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "renders a link rather than a player" in report.warnings[0]


def test_check_accepts_a_video_on_its_own_line(tmp_path: Path) -> None:
    report = check(tmp_path, "the repro:\n\n![repro]({p})\n", "clip.mp4")
    assert report.errors == []
    assert report.warnings == []


def test_check_warns_about_a_symlink(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (tmp_path / "environ").write_text("GH_TOKEN=supersecret")
    (media / "shot.png").symlink_to(tmp_path / "environ")

    report = embed_media.check_media(media, ["a prose-only finding"])
    assert report.warnings == ["shot.png: a symlink, so it is never uploaded"]


def test_check_reports_a_repeated_bad_citation_once(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    cite = f"{media / 'missing.png'}"
    report = embed_media.check_media(media, [f"![a]({cite})", f"![b]({cite})"])
    assert len(report.errors) == 1


def test_check_agrees_with_what_the_upload_would_rewrite(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    body = f"![the bug]({media / 'shot.png'})"
    assert embed_media.check_media(media, [body]).errors == []
    assert substitute(body, {str(media / "shot.png"): URL}) == f"![the bug]({URL})"


def test_cli_check_exits_zero_and_uploads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    body = f"![the bug]({media / 'shot.png'})"
    with (
        mock.patch.object(embed_media, "resolve_github_token") as resolver,
        mock.patch.object(embed_media, "upload_asset") as uploader,
    ):
        out, err = run_cli(build_check_args(media), body, monkeypatch, capsys)

    resolver.assert_not_called()
    uploader.assert_not_called()
    assert out == ""
    assert "OK: 1 media reference(s) resolve" in err


@pytest.mark.parametrize("cite", ["shto.png", "shot.png", "./shot.png"])
def test_cli_check_exits_nonzero_on_an_unresolvable_citation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cite: str,
) -> None:
    media = make_media(tmp_path)
    text = f"![the bug]({media / cite})" if cite == "shto.png" else f"![the bug]({cite})"
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    args = build_check_args(media)
    with pytest.raises(SystemExit, match="^1$"):
        args.func(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ERROR: input cites media that will not render" in captured.err


def test_cli_check_reports_warnings_without_uploading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path, "clip.mp4")
    (media / "unused.png").write_bytes(b"\x89PNG")
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        out, err = run_cli(
            build_check_args(media),
            f"see [repro]({media / 'clip.mp4'}) inline",
            monkeypatch,
            capsys,
        )
    uploader.assert_not_called()
    assert out == ""
    assert "renders a link rather than a player" in err
    assert "unused.png: cited by nothing" in err


def test_cli_check_reports_unsupported_and_missing_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path, "notes.txt")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(f"[notes]({media / 'notes.txt'})\n![missing]({media / 'missing.png'})"),
    )
    args = build_check_args(media)
    with pytest.raises(SystemExit, match="^1$"):
        args.func(args)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported extension" in captured.err
    assert "no such file" in captured.err


def test_cli_strips_a_citation_naming_no_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        out, err = run_cli(
            build_args(media), f"evidence: ![the bug]({media / 'shto.png'})", monkeypatch, capsys
        )

    uploader.assert_not_called()
    assert out == "evidence: the bug"
    assert "no such capture, stripping" in err


def test_cli_strips_a_typo_while_still_embedding_the_capture_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    monkeypatch.setenv("GH_TOKEN", "t")
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        out, _ = run_cli(
            build_args(media),
            f"![ok]({media / 'shot.png'}) and ![typo]({media / 'shto.png'})",
            monkeypatch,
            capsys,
        )

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert out == f"![ok]({URL}) and typo"


@pytest.mark.parametrize("cite", ["shot.png", "./shot.png"])
def test_cli_strips_a_bare_filename_citation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cite: str,
) -> None:
    media = make_media(tmp_path)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        out, _ = run_cli(build_args(media), f"evidence: ![the bug]({cite})", monkeypatch, capsys)

    uploader.assert_not_called()
    assert out == "evidence: the bug"


def test_cli_leaves_a_link_outside_the_media_directory_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    body = "see [the icon](docs/static/img/shot.png)"
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        out, _ = run_cli(build_args(media), body, monkeypatch, capsys)

    uploader.assert_not_called()
    assert out == body


def test_cli_check_tolerates_a_missing_media_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, err = run_cli(
        build_check_args(tmp_path / "absent"), "a prose-only finding", monkeypatch, capsys
    )
    assert out == ""
    assert "OK: 0 media reference(s) resolve" in err


def test_cli_requires_a_repository_id_without_check(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    args = build_parser().parse_args([
        "embed-media",
        "--dir",
        str(media),
    ])
    with pytest.raises(SystemExit, match="^2$"):
        args.func(args)
