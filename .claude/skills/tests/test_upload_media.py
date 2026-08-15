import json
from pathlib import Path

import pytest
from skills.cli import build_parser
from skills.commands import upload_media
from skills.commands.upload_media import rewrite_payload, substitute

URL = "https://github.com/user-attachments/assets/2f1c0a3e-0000-4000-8000-000000000001"
URLS = {"shot.png": URL}


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


def test_rewrite_payload_covers_body_and_inline_comments() -> None:
    payload = {
        "event": "COMMENT",
        "body": "Race shown in `shot.png`\n\n🤖 Generated with Claude",
        "comments": [{"path": "a.py", "body": "🔴 **CRITICAL:** ![x](shot.png)", "line": 1}],
    }
    result = rewrite_payload(payload, URLS)
    assert result["body"] == f"Race shown in [`shot.png`]({URL})\n\n🤖 Generated with Claude"
    assert result["comments"][0]["body"] == f"🔴 **CRITICAL:** ![x]({URL})"


def test_rewrite_payload_preserves_the_trailing_footer() -> None:
    payload = {"body": "`shot.png`\n\n🤖 Generated with Claude", "comments": []}
    assert rewrite_payload(payload, URLS)["body"].endswith("🤖 Generated with Claude")


def test_rewrite_payload_tolerates_missing_and_malformed_fields() -> None:
    assert rewrite_payload({}, URLS) == {}
    assert rewrite_payload({"body": None, "comments": None}, URLS) == {
        "body": None,
        "comments": None,
    }


def run_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    media = tmp_path / "media"
    media.mkdir(exist_ok=True)
    (media / "shot.png").write_bytes(b"\x89PNG")
    monkeypatch.setattr(upload_media, "upload_asset", lambda *a, **k: URL)
    args = build_parser().parse_args([
        "upload-media",
        "--dir",
        str(media),
        "--target",
        str(target),
        "--repository-id",
        "136202695",
        "--token",
        "t",
    ])
    args.func(args)


def test_cli_rewrites_a_json_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "review-payload.json"
    target.write_text(json.dumps({"body": "`shot.png`", "comments": []}))
    run_cli(tmp_path, monkeypatch, target)
    assert json.loads(target.read_text())["body"] == f"[`shot.png`]({URL})"


def test_cli_rewrites_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "body.md"
    target.write_text("![alt](shot.png)")
    run_cli(tmp_path, monkeypatch, target)
    assert target.read_text() == f"![alt]({URL})"


def test_cli_leaves_the_target_alone_when_every_upload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "shot.png").write_bytes(b"\x89PNG")
    target = tmp_path / "review-payload.json"
    original = json.dumps({"body": "`shot.png`", "comments": []})
    target.write_text(original)

    monkeypatch.setattr(upload_media, "upload_asset", lambda *a, **k: None)
    args = build_parser().parse_args([
        "upload-media",
        "--dir",
        str(media),
        "--target",
        str(target),
        "--repository-id",
        "136202695",
        "--token",
        "t",
    ])
    args.func(args)
    assert target.read_text() == original


def test_cli_is_a_noop_when_there_is_no_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "empty"
    media.mkdir()
    target = tmp_path / "body.md"
    target.write_text("unchanged")
    args = build_parser().parse_args([
        "upload-media",
        "--dir",
        str(media),
        "--target",
        str(target),
        "--repository-id",
        "136202695",
        "--token",
        "t",
    ])
    args.func(args)
    assert target.read_text() == "unchanged"


def test_upload_asset_skips_an_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("x")
    assert upload_media.upload_asset(path, "1", "t") is None


def test_upload_asset_skips_a_file_over_the_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(upload_media, "MAX_BYTES", 4)
    path = tmp_path / "big.png"
    path.write_bytes(b"12345")
    assert upload_media.upload_asset(path, "1", "t") is None
