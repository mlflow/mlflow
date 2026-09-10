import argparse
import json
from pathlib import Path
from unittest import mock

import pytest
from skills.cli import build_parser
from skills.commands import embed_media
from skills.commands.embed_media import rewrite_payload, substitute
from skills.github import uploads

URL = "https://github.com/user-attachments/assets/2f1c0a3e-0000-4000-8000-000000000001"
MEDIA = "/tmp/review-media"
SHOT = f"{MEDIA}/shot.png"
CLIP = f"{MEDIA}/clip.mp4"
URLS = {SHOT: URL}


def build_args(media: Path, target: Path) -> argparse.Namespace:
    return build_parser().parse_args([
        "embed-media",
        "--dir",
        str(media),
        "--target",
        str(target),
        "--repository-id",
        "136202695",
    ])


def build_check_args(media: Path, target: Path) -> argparse.Namespace:
    return build_parser().parse_args([
        "embed-media",
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


def check(tmp_path: Path, body: str, name: str = "shot.png") -> embed_media.CheckReport:
    media = make_media(tmp_path, name)
    return embed_media.check_media(media, [body.format(p=media / name, media=media)])


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


def test_rewrite_payload_covers_body_and_inline_comments() -> None:
    payload = {
        "event": "COMMENT",
        "body": f"Race shown in [the trace]({SHOT})\n\nNothing else stood out.",
        "comments": [{"path": "a.py", "body": f"🔴 **CRITICAL:** ![x]({SHOT})", "line": 1}],
    }
    result = rewrite_payload(payload, URLS)
    assert result["body"] == f"Race shown in [the trace]({URL})\n\nNothing else stood out."
    assert result["comments"][0]["body"] == f"🔴 **CRITICAL:** ![x]({URL})"


def test_rewrite_payload_neutralizes_unavailable_media_in_comments() -> None:
    payload = {"body": "b", "comments": [{"body": f"![the bug]({SHOT})"}]}
    assert rewrite_payload(payload, {}, [SHOT])["comments"][0]["body"] == "the bug"


def test_rewrite_payload_preserves_trailing_text() -> None:
    payload = {"body": f"![x]({SHOT})\n\nNothing else stood out.", "comments": []}
    assert rewrite_payload(payload, URLS)["body"].endswith("Nothing else stood out.")


def test_rewrite_payload_tolerates_missing_and_malformed_fields() -> None:
    assert rewrite_payload({}, URLS) == {}
    assert rewrite_payload({"body": None, "comments": None}, URLS) == {
        "body": None,
        "comments": None,
    }


def test_cli_rewrites_a_json_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "review-payload.json"
    target.write_text(json.dumps({"body": f"[the trace]({media / 'shot.png'})", "comments": []}))

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert json.loads(target.read_text())["body"] == f"[the trace]({URL})"


def test_cli_rewrites_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"![alt]({media / 'shot.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert target.read_text() == f"![alt]({URL})"


def test_cli_degrades_to_prose_when_every_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"evidence: ![the bug]({media / 'shot.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(
        embed_media, "upload_asset", side_effect=uploads.UploadFailed("shot.png: boom")
    ) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert target.read_text() == "evidence: the bug"


def test_cli_never_posts_a_local_path_when_no_token_resolves(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"evidence: ![the bug]({media / 'shot.png'})")

    args = build_args(media, target)
    with (
        mock.patch.object(embed_media, "resolve_github_token", return_value=None) as resolver,
        mock.patch.object(embed_media, "upload_asset") as uploader,
    ):
        args.func(args)

    resolver.assert_called_once()
    uploader.assert_not_called()
    assert target.read_text() == "evidence: the bug"


def test_cli_is_a_noop_when_there_is_no_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "empty"
    media.mkdir()
    target = tmp_path / "body.md"
    target.write_text("unchanged")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "unchanged"


def test_cli_never_reads_a_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = tmp_path / "media"
    media.mkdir()
    secret = tmp_path / "environ"
    secret.write_text("GH_TOKEN=supersecret")
    (media / "shot.png").symlink_to(secret)
    target = tmp_path / "body.md"
    target.write_text(f"![the secret]({media / 'shot.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_not_called()
    # Never uploaded, so the citation resolves to nothing and must not post the path.
    assert target.read_text() == "the secret"


def test_cli_annotates_once_and_degrades_when_the_token_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_media(tmp_path)
    (media / "second.png").write_bytes(b"\x89PNG")
    target = tmp_path / "body.md"
    # Both are cited, so a second upload would be attempted if the loop kept going.
    target.write_text(
        f"evidence: ![the bug]({media / 'shot.png'}) and ![more]({media / 'second.png'})"
    )

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(
        embed_media,
        "upload_asset",
        side_effect=uploads.UploadFailed("the credential was rejected (401)", status=401),
    ) as uploader:
        args.func(args)

    # One annotation for the run, and the loop stops rather than retrying a dead token.
    out = capsys.readouterr().out
    assert out.count("::warning::media upload stopped: the credential was rejected (401)") == 1
    assert uploader.call_count == 1
    assert target.read_text() == "evidence: the bug and more"


def test_cli_uploads_only_media_the_target_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "cited.png")
    (media / "scratch.png").write_bytes(b"\x89PNG")
    target = tmp_path / "body.md"
    target.write_text(f"evidence: ![the bug]({media / 'cited.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "cited.png", "136202695", "t")
    assert target.read_text() == f"evidence: ![the bug]({URL})"


def test_cli_uploads_nothing_when_no_media_is_referenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "scratch.png")
    target = tmp_path / "body.md"
    target.write_text("a prose-only finding")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "a prose-only finding"


def test_cli_uploads_media_referenced_by_an_inline_comment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "cited.png")
    (media / "scratch.png").write_bytes(b"\x89PNG")
    target = tmp_path / "review-payload.json"
    target.write_text(
        json.dumps({"body": "b", "comments": [{"body": f"see [it]({media / 'cited.png'})"}]})
    )

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "cited.png", "136202695", "t")
    assert json.loads(target.read_text())["comments"][0]["body"] == f"see [it]({URL})"


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path, "longshot.png")
    (media / "shot.png").write_bytes(b"\x89PNG")
    target = tmp_path / "body.md"
    target.write_text(f"![x]({media / 'longshot.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "longshot.png", "136202695", "t")
    assert target.read_text() == f"![x]({URL})"


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


def test_check_reads_the_body_and_comments_of_a_json_payload(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "review-payload.json"
    target.write_text(
        json.dumps({
            "body": f"![overview]({media / 'shot.png'})",
            "comments": [{"body": f"and ![again]({media / 'missing.png'})"}],
        })
    )
    report = embed_media.check_media(media, embed_media.target_bodies(target))
    assert report.cited == ["shot.png"]
    assert len(report.errors) == 1
    assert "missing.png): no such file" in report.errors[0]


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    body = f"![the bug]({media / 'shot.png'})"
    target.write_text(body)

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_check_args(media, target)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == body


def test_cli_check_exits_nonzero_on_an_unresolvable_citation(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"![the bug]({media / 'shto.png'})")

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


def test_cli_strips_a_citation_naming_no_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"evidence: ![the bug]({media / 'shto.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "evidence: the bug"


def test_cli_strips_a_typo_while_still_embedding_the_capture_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"![ok]({media / 'shot.png'}) and ![typo]({media / 'shto.png'})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset", return_value=URL) as uploader:
        args.func(args)

    uploader.assert_called_once_with(media / "shot.png", "136202695", "t")
    assert target.read_text() == f"![ok]({URL}) and typo"


@pytest.mark.parametrize("cite", ["shot.png", "./shot.png"])
def test_cli_strips_a_bare_filename_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cite: str
) -> None:
    # The form this PR stopped teaching: nothing resolves it, so it must not post.
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"evidence: ![the bug]({cite})")

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == "evidence: the bug"


def test_cli_leaves_a_link_outside_the_media_directory_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    body = "see [the icon](docs/static/img/shot.png)"
    target.write_text(body)

    monkeypatch.setenv("GH_TOKEN", "t")
    args = build_args(media, target)
    with mock.patch.object(embed_media, "upload_asset") as uploader:
        args.func(args)

    uploader.assert_not_called()
    assert target.read_text() == body


def test_cli_check_tolerates_a_missing_media_directory(tmp_path: Path) -> None:
    target = tmp_path / "body.md"
    target.write_text("a prose-only finding")
    args = build_check_args(tmp_path / "absent", target)
    args.func(args)


def test_cli_requires_a_repository_id_without_check(tmp_path: Path) -> None:
    media = make_media(tmp_path)
    target = tmp_path / "body.md"
    target.write_text(f"![the bug]({media / 'shot.png'})")

    args = build_parser().parse_args([
        "embed-media",
        "--dir",
        str(media),
        "--target",
        str(target),
    ])
    with pytest.raises(SystemExit, match="^2$"):
        args.func(args)
