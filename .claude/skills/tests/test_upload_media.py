import argparse
import http.client
import io
import json
import urllib.error
from pathlib import Path
from unittest import mock

import pytest
from skills.cli import build_parser
from skills.commands import upload_media
from skills.commands.upload_media import rewrite_payload, substitute

URL = "https://github.com/user-attachments/assets/2f1c0a3e-0000-4000-8000-000000000001"
URLS = {"shot.png": URL}


def build_args(media: Path, target: Path) -> argparse.Namespace:
    return build_parser().parse_args([
        "upload-media",
        "--dir",
        str(media),
        "--target",
        str(target),
        "--repository-id",
        "136202695",
    ])


def build_check_args(media: Path, target: Path) -> argparse.Namespace:
    return build_parser().parse_args([
        "upload-media",
        "--dir",
        str(media),
        "--target",
        str(target),
        "--check",
    ])


def make_media(tmp_path: Path, name: str = "shot.png") -> Path:
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    (media / name).write_bytes(b"\x89PNG")
    return media


def check(tmp_path: Path, body: str, name: str = "shot.png") -> upload_media.CheckReport:
    return upload_media.check_media(make_media(tmp_path, name), [body])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![alt](shot.png)", f"![alt]({URL})"),
        ("![alt](./shot.png)", f"![alt]({URL})"),
        ("[link](shot.png)", f"[link]({URL})"),
        ("see `shot.png` here", f"see [`shot.png`]({URL}) here"),
        ("nothing to do", "nothing to do"),
        ("other.png stays", "other.png stays"),
    ],
)
def test_substitute_rewrites_each_reference_form(text: str, expected: str) -> None:
    assert substitute(text, URLS) == expected


def test_substitute_is_idempotent() -> None:
    once = substitute("see `shot.png`", URLS)
    assert substitute(once, URLS) == once


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![repro](clip.mp4)", f"\n{URL}\n"),
        ("`clip.mp4`", f"\n{URL}\n"),
        ("before\n![repro](clip.mp4)\nafter", f"before\n\n{URL}\n\nafter"),
        ("see `clip.mp4` inline", f"see [`clip.mp4`]({URL}) inline"),
        # ![]() around a video URL renders as a broken image, so it must become a link.
        ("see ![repro](clip.mp4) inline", f"see [repro]({URL}) inline"),
    ],
)
def test_substitute_promotes_a_standalone_video_to_a_bare_url(text: str, expected: str) -> None:
    assert substitute(text, {"clip.mp4": URL}) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![a screenshot](shot.png)", "a screenshot"),
        ("[a screenshot](shot.png)", "a screenshot"),
        ("![](shot.png)", "shot.png"),
        ("see `shot.png`", "see `shot.png`"),
    ],
)
def test_substitute_strips_markup_for_media_that_never_uploaded(text: str, expected: str) -> None:
    assert substitute(text, {}, ["shot.png"]) == expected


def test_rewrite_payload_covers_body_and_inline_comments() -> None:
    payload = {
        "event": "COMMENT",
        "body": "Race shown in `shot.png`\n\n🤖 Generated with Claude",
        "comments": [{"path": "a.py", "body": "🔴 **CRITICAL:** ![x](shot.png)", "line": 1}],
    }
    result = rewrite_payload(payload, URLS)
    assert result["body"] == f"Race shown in [`shot.png`]({URL})\n\n🤖 Generated with Claude"
    assert result["comments"][0]["body"] == f"🔴 **CRITICAL:** ![x]({URL})"


def test_rewrite_payload_neutralizes_unavailable_media_in_comments() -> None:
    payload = {"body": "b", "comments": [{"body": "![the bug](shot.png)"}]}
    assert rewrite_payload(payload, {}, ["shot.png"])["comments"][0]["body"] == "the bug"


def test_rewrite_payload_preserves_the_trailing_footer() -> None:
    payload = {"body": "`shot.png`\n\n🤖 Generated with Claude", "comments": []}
    assert rewrite_payload(payload, URLS)["body"].endswith("🤖 Generated with Claude")


def test_rewrite_payload_tolerates_missing_and_malformed_fields() -> None:
    assert rewrite_payload({}, URLS) == {}
    assert rewrite_payload({"body": None, "comments": None}, URLS) == {
        "body": None,
        "comments": None,
    }


def test_cli_rewrites_a_json_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "review-payload.json"
    target.write_text(json.dumps({"body": "`shot.png`", "comments": []}))

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert json.loads(target.read_text())["body"] == f"[`shot.png`]({URL})"


def test_cli_rewrites_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text("![alt](shot.png)")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert target.read_text() == f"![alt]({URL})"


def test_cli_degrades_to_prose_when_every_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text("evidence: ![the bug](shot.png)")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=None) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert target.read_text() == "evidence: the bug"


def test_cli_degrades_to_prose_when_the_token_env_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text("evidence: ![the bug](shot.png) and `shot.png`")

    monkeypatch.delenv(upload_media.TOKEN_ENV, raising=False)
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "evidence: the bug and `shot.png`"


def test_cli_is_a_noop_when_there_is_no_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "empty"
    media.mkdir()
    target = tmp_path / "body.md"
    target.write_text("unchanged")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "unchanged"


def test_cli_never_reads_a_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    secret = tmp_path / "environ"
    secret.write_text("UPLOAD_MEDIA_TOKEN=supersecret")
    (media / "shot.png").symlink_to(secret)
    target = tmp_path / "body.md"
    target.write_text("`shot.png`")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "`shot.png`"


def test_upload_asset_builds_the_request_and_returns_the_url(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    response = io.BytesIO(json.dumps({"url": URL}).encode())
    with mock.patch("urllib.request.urlopen", return_value=response) as opener:
        assert upload_media.upload_asset(path, "136202695", "tok") == URL

    request = opener.call_args.args[0]
    assert "name=shot.png" in request.full_url
    assert "content_type=image%2Fpng" in request.full_url
    assert "repository_id=136202695" in request.full_url
    assert request.get_header("Authorization") == "Bearer tok"
    assert request.data == b"\x89PNG"


@pytest.mark.parametrize(
    "outcome",
    [
        urllib.error.URLError("boom"),
        TimeoutError("slow"),
        # Escapes URLError: urlopen only wraps h.request(), not h.getresponse().
        http.client.RemoteDisconnected("closed"),
        http.client.IncompleteRead(b"partial"),
        json.JSONDecodeError("bad", "", 0),
    ],
)
def test_upload_asset_returns_none_when_the_request_fails(
    tmp_path: Path, outcome: Exception
) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with mock.patch("urllib.request.urlopen", side_effect=outcome) as opener:
        assert upload_media.upload_asset(path, "1", "t") is None

    opener.assert_called_once()


def test_upload_asset_returns_none_when_the_response_carries_no_url(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"{}")) as opener:
        assert upload_media.upload_asset(path, "1", "t") is None

    opener.assert_called_once()


def test_upload_asset_skips_an_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("x")
    assert upload_media.upload_asset(path, "1", "t") is None


def test_upload_asset_skips_a_file_over_the_size_cap(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(b"12345")
    with mock.patch.object(upload_media, "MAX_IMAGE_BYTES", 4):
        assert upload_media.upload_asset(path, "1", "t") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shot.png", upload_media.MAX_IMAGE_BYTES),
        ("shot.gif", upload_media.MAX_IMAGE_BYTES),
        ("clip.mp4", upload_media.MAX_VIDEO_BYTES),
        ("clip.MOV", upload_media.MAX_VIDEO_BYTES),
        ("clip.webm", upload_media.MAX_VIDEO_BYTES),
    ],
)
def test_max_bytes_is_larger_for_video(name: str, expected: int) -> None:
    assert upload_media.max_bytes(name) == expected


def test_a_video_between_the_image_and_video_caps_is_not_skipped(tmp_path: Path) -> None:
    # The old single 10MB cap silently dropped exactly this: a screen recording.
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"0" * (upload_media.MAX_IMAGE_BYTES + 1))

    response = io.BytesIO(json.dumps({"url": URL}).encode())
    with mock.patch("urllib.request.urlopen", return_value=response):
        assert upload_media.upload_asset(path, "1", "t") == URL


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(upload_media.UPLOAD_URL, code, "nope", {}, None)  # type: ignore[arg-type]


def test_upload_asset_raises_on_a_rejected_token(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with (
        mock.patch("urllib.request.urlopen", side_effect=http_error(401)),
        pytest.raises(upload_media.TokenRejected, match="rejected \\(401\\)"),
    ):
        upload_media.upload_asset(path, "1", "t")


def test_upload_asset_returns_none_on_other_http_errors(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(b"\x89PNG")

    with mock.patch("urllib.request.urlopen", side_effect=http_error(500)):
        assert upload_media.upload_asset(path, "1", "t") is None


def test_cli_annotates_once_and_degrades_when_the_token_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    (media / "second.png").write_bytes(b"\x89PNG")
    target = tmp_path / "body.md"
    target.write_text("evidence: ![the bug](shot.png)")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(
        upload_media,
        "upload_asset",
        side_effect=upload_media.TokenRejected("UPLOAD_MEDIA_TOKEN was rejected (401)"),
    ) as uploader:
        args.func(args)

    # One annotation for the run, and the loop stops rather than retrying a dead token.
    out = capsys.readouterr().out
    assert out.count(f"::warning::{upload_media.TOKEN_ENV} was rejected (401)") == 1
    assert uploader.call_count == 1
    assert target.read_text() == "evidence: the bug"


def test_cli_uploads_only_media_the_target_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "cited.png")
    (media / "scratch.png").write_bytes(b"\x89PNG")
    target = tmp_path / "body.md"
    target.write_text("evidence: ![the bug](cited.png)")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "cited.png", "136202695", "t")
    assert target.read_text() == f"evidence: ![the bug]({URL})"


def test_cli_uploads_nothing_when_no_media_is_referenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "scratch.png")
    target = tmp_path / "body.md"
    target.write_text("a prose-only finding")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "a prose-only finding"


def test_cli_uploads_media_referenced_by_an_inline_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "cited.png")
    (media / "scratch.png").write_bytes(b"\x89PNG")
    target = tmp_path / "review-payload.json"
    target.write_text(json.dumps({"body": "b", "comments": [{"body": "see `cited.png`"}]}))

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "cited.png", "136202695", "t")
    assert json.loads(target.read_text())["comments"][0]["body"] == f"see [`cited.png`]({URL})"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("![x](shot.png)", True),
        ("![x](./shot.png)", True),
        ("see `shot.png`", True),
        # A name that is only a suffix of a cited one is not a reference.
        ("![x](longshot.png)", False),
        ("shot.png bare mention", False),
        ("nothing here", False),
    ],
)
def test_is_referenced_matches_only_real_reference_forms(text: str, expected: bool) -> None:
    assert upload_media.is_referenced("shot.png", text) is expected


@pytest.mark.parametrize(
    "body",
    [
        "![the bug](shot.png)",
        "![the bug](./shot.png)",
        "[the bug](shot.png)",
        "see `shot.png`",
    ],
)
def test_check_accepts_every_form_the_rewriter_understands(tmp_path: Path, body: str) -> None:
    report = check(tmp_path, body)
    assert report.errors == []
    assert report.warnings == []
    assert report.cited == ["shot.png"]


@pytest.mark.parametrize("body", ["![the bug](shto.png)", "![the bug](./shto.png)"])
def test_check_rejects_a_citation_naming_no_captured_file(tmp_path: Path, body: str) -> None:
    report = check(tmp_path, body)
    assert len(report.errors) == 1
    assert "no such file" in report.errors[0]


def test_check_rejects_a_citation_carrying_a_path(tmp_path: Path) -> None:
    report = check(tmp_path, "![the bug](media/shot.png)")
    assert len(report.errors) == 1
    assert "cite shot.png by bare filename" in report.errors[0]
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
    assert upload_media.check_media(media, [body]).errors == []


def test_check_warns_about_a_capture_nothing_cites(tmp_path: Path) -> None:
    report = check(tmp_path, "a prose-only finding")
    assert report.errors == []
    assert report.warnings == ["shot.png: cited by nothing, so it is not uploaded"]


def test_check_rejects_a_cited_file_with_an_unsupported_extension(tmp_path: Path) -> None:
    report = check(tmp_path, "see `notes.txt`", "notes.txt")
    assert report.errors == ["notes.txt: unsupported extension, so the reference is dropped"]


def test_check_rejects_a_cited_file_over_the_size_cap(tmp_path: Path) -> None:
    with mock.patch.object(upload_media, "MAX_IMAGE_BYTES", 2):
        report = check(tmp_path, "![the bug](shot.png)")
    assert len(report.errors) == 1
    assert "exceeds the 2 byte cap" in report.errors[0]


def test_check_warns_when_a_video_is_cited_mid_paragraph(tmp_path: Path) -> None:
    report = check(tmp_path, "the repro is `clip.mp4` here", "clip.mp4")
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "renders a link rather than a player" in report.warnings[0]


def test_check_accepts_a_video_on_its_own_line(tmp_path: Path) -> None:
    report = check(tmp_path, "the repro:\n\n![repro](clip.mp4)\n", "clip.mp4")
    assert report.errors == []
    assert report.warnings == []


def test_check_warns_about_a_symlink(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (tmp_path / "environ").write_text("UPLOAD_MEDIA_TOKEN=supersecret")
    (media / "shot.png").symlink_to(tmp_path / "environ")

    report = upload_media.check_media(media, ["a prose-only finding"])
    assert report.warnings == ["shot.png: a symlink, so it is never uploaded"]


def test_check_reads_the_body_and_comments_of_a_json_payload(tmp_path: Path) -> None:
    target = tmp_path / "review-payload.json"
    target.write_text(
        json.dumps({
            "body": "![overview](shot.png)",
            "comments": [{"body": "and ![again](missing.png)"}],
        })
    )
    report = upload_media.check_media(make_media(tmp_path), upload_media.target_bodies(target))
    assert report.cited == ["shot.png"]
    assert len(report.errors) == 1
    assert "(missing.png): no such file" in report.errors[0]


def test_check_reports_a_repeated_bad_citation_once(tmp_path: Path) -> None:
    report = upload_media.check_media(
        make_media(tmp_path), ["![a](missing.png)", "![b](missing.png)"]
    )
    assert len(report.errors) == 1


def test_check_agrees_with_what_the_upload_would_rewrite(tmp_path: Path) -> None:
    body = "![the bug](shot.png)"
    assert check(tmp_path, body).errors == []
    assert substitute(body, URLS) != body


def test_cli_check_exits_zero_and_uploads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text("![the bug](shot.png)")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_check_args(media, target)
    with mock.patch.object(upload_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "![the bug](shot.png)"


def test_cli_check_exits_nonzero_on_an_unresolvable_citation(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text("![the bug](shto.png)")

    args = build_check_args(media, target)
    with pytest.raises(SystemExit, match="^1$"):
        args.func(args)


@pytest.mark.parametrize("contents", ["not json", '{"body": '])
def test_cli_check_exits_nonzero_on_a_malformed_json_payload(tmp_path: Path, contents: str) -> None:
    target = tmp_path / "review-payload.json"
    target.write_text(contents)

    args = build_check_args(make_media(tmp_path), target)
    with pytest.raises(SystemExit, match="^1$"):
        args.func(args)


def test_cli_check_exits_nonzero_when_the_target_is_missing(tmp_path: Path) -> None:
    args = build_check_args(make_media(tmp_path), tmp_path / "absent.md")
    with pytest.raises(SystemExit, match="^1$"):
        args.func(args)


def test_cli_check_tolerates_a_missing_media_directory(tmp_path: Path) -> None:
    target = tmp_path / "body.md"
    target.write_text("a prose-only finding")
    args = build_check_args(tmp_path / "absent", target)
    args.func(args)


def test_cli_requires_a_repository_id_without_check(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text("![the bug](shot.png)")

    args = build_parser().parse_args([
        "upload-media",
        "--dir",
        str(media),
        "--target",
        str(target),
    ])
    with pytest.raises(SystemExit, match="^2$"):
        args.func(args)


def test_cli_does_not_upload_a_name_that_is_a_suffix_of_a_cited_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "longshot.png")
    (media / "shot.png").write_bytes(b"\x89PNG")
    target = tmp_path / "body.md"
    target.write_text("![x](longshot.png)")

    monkeypatch.setenv(upload_media.TOKEN_ENV, "t")
    args = build_args(media, target)
    with mock.patch.object(upload_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "longshot.png", "136202695", "t")
    assert target.read_text() == f"![x]({URL})"
