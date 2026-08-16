# ruff: noqa: T201
"""Upload review media to GitHub's user-attachments store and swap filenames for URLs."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UPLOAD_URL = "https://uploads.github.com/user-attachments/assets"
# Read from the environment, never argv: a PAT in a CLI argument is visible in the
# process list for the life of the call.
TOKEN_ENV = "UPLOAD_MEDIA_TOKEN"

# The name's extension must agree with content_type or the endpoint returns 422.
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}


class TokenRejected(Exception):
    """Raised on 401: the credential is dead, so every remaining upload would fail too."""


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def max_bytes(name: str) -> int:
    return MAX_VIDEO_BYTES if is_video(name) else MAX_IMAGE_BYTES


def upload_asset(path: Path, repository_id: str, token: str) -> str | None:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        print(f"  skip {path.name}: unsupported extension", file=sys.stderr)
        return None

    size = path.stat().st_size
    if size > (limit := max_bytes(path.name)):
        print(f"  skip {path.name}: {size} bytes exceeds {limit}", file=sys.stderr)
        return None

    query = urllib.parse.urlencode({
        "name": path.name,
        "content_type": mime,
        "repository_id": repository_id,
    })
    request = urllib.request.Request(
        f"{UPLOAD_URL}?{query}",
        data=path.read_bytes(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            body = json.load(resp)
    # Must precede OSError: HTTPError is a URLError, which is an OSError.
    except urllib.error.HTTPError as e:
        print(f"  failed {path.name}: {e}", file=sys.stderr)
        if e.code == 401:
            raise TokenRejected(f"{TOKEN_ENV} was rejected (401); it may have expired") from e
        return None
    except (OSError, http.client.HTTPException, ValueError) as e:
        print(f"  failed {path.name}: {e}", file=sys.stderr)
        return None

    match body:
        case {"url": str(url)} if url:
            print(f"  uploaded {path.name} -> {url}")
            return url
        case _:
            print(f"  failed {path.name}: response carried no url", file=sys.stderr)
            return None


def link_target(name: str) -> str:
    return rf"\]\((?:\./)?{re.escape(name)}\)"


def code_span(name: str) -> str:
    return rf"`{re.escape(name)}`"


def reference_pattern(name: str) -> str:
    return f"{link_target(name)}|{code_span(name)}"


def is_referenced(name: str, text: str) -> bool:
    return re.search(reference_pattern(name), text) is not None


def standalone_pattern(name: str) -> str:
    # GitHub renders a video player only for a bare URL alone in its own paragraph.
    return rf"(?m)^[ \t]*(?:!?\[[^\]]*{link_target(name)}|{code_span(name)})[ \t]*$"


def substitute(text: str, urls: dict[str, str], unavailable: Iterable[str] = ()) -> str:
    for name, url in urls.items():
        link = link_target(name)
        code = code_span(name)
        if is_video(name):
            # Promote a reference that already sits on its own line.
            text = re.sub(standalone_pattern(name), f"\n{url}\n", text)
            # Whatever is left is mid-sentence, where ![]() around a video URL renders
            # as a broken image. Drop the bang so it degrades to a link instead.
            text = re.sub(rf"!(?=\[[^\]]*{link})", "", text)
        text = re.sub(link, f"]({url})", text)
        # Lookarounds keep a second pass a no-op.
        text = re.sub(rf"(?<!\[){code}(?!\])", f"[`{name}`]({url})", text)

    # A name with no URL (upload failed, or the file was skipped) would otherwise
    # survive as ![desc](shot.png), which GitHub resolves repo-relative and renders
    # as a broken image. Strip the markup so it degrades to prose.
    for name in unavailable:
        link = link_target(name)
        text = re.sub(rf"!?\[{link}", name, text)
        text = re.sub(rf"!?\[([^\]]+){link}", r"\1", text)
    return text


def collect_files(directory: Path) -> tuple[list[Path], list[str]]:
    """Return the uploadable files, and the names rejected for being symlinks.

    ``is_file()`` follows symlinks, so a link planted here (say secret.png ->
    /proc/self/environ) would publish this process's own UPLOAD_MEDIA_TOKEN as an
    attachment. Claude writes this directory, and a poisoned diff steers Claude.
    """
    files: list[Path] = []
    symlinks: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            symlinks.append(path.name)
        elif path.is_file():
            files.append(path)
    return files, symlinks


def rewrite_payload(
    payload: dict[str, Any], urls: dict[str, str], unavailable: Iterable[str] = ()
) -> dict[str, Any]:
    # The schema pins body to end with the Claude footer, so only substitute, never append.
    match payload:
        case {"body": str(body)}:
            payload["body"] = substitute(body, urls, unavailable)
    match payload:
        case {"comments": [*comments]}:
            for comment in comments:
                match comment:
                    case {"body": str(body)}:
                        comment["body"] = substitute(body, urls, unavailable)
    return payload


LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)


def iter_bodies(payload: dict[str, Any]) -> Iterator[str]:
    match payload:
        case {"body": str(body)}:
            yield body
    match payload:
        case {"comments": [*comments]}:
            for comment in comments:
                match comment:
                    case {"body": str(body)}:
                        yield body


def target_bodies(target: Path) -> list[str]:
    text = target.read_text()
    if target.suffix != ".json":
        return [text]
    return list(iter_bodies(json.loads(text)))


def check_media(directory: Path, texts: list[str]) -> CheckReport:
    """Report what the upload would silently drop or leave broken in ``texts``."""
    report = CheckReport()
    files, symlinks = collect_files(directory) if directory.is_dir() else ([], [])
    by_name = {p.name: p for p in files}
    report.warnings += [f"{name}: a symlink, so it is never uploaded" for name in symlinks]

    cited = {name for name in by_name if any(is_referenced(name, t) for t in texts)}

    for text in texts:
        for raw in LINK.findall(text):
            if "://" in raw:
                continue
            path = raw.removeprefix("./")
            name = path.rsplit("/", 1)[-1]
            if name in by_name:
                cited.add(name)
                if path != name:
                    report.errors.append(
                        f"({raw}): cite {name} by bare filename, or the path is left alone "
                        "and GitHub resolves it against the repository"
                    )
            # A path still resolves against the repository, so only a bare media filename
            # that matches no capture is certain to render as a broken image.
            elif path == name and Path(name).suffix.lower() in MIME_TYPES:
                report.errors.append(
                    f"({raw}): no such file in {directory}, so the reference ships as-is "
                    "and GitHub renders it as a broken image"
                )

    for name in sorted(cited):
        if Path(name).suffix.lower() not in MIME_TYPES:
            report.errors.append(f"{name}: unsupported extension, so the reference is dropped")
        elif (size := by_name[name].stat().st_size) > (limit := max_bytes(name)):
            report.errors.append(
                f"{name}: {size} bytes exceeds the {limit} byte cap, so the reference is dropped"
            )
        elif is_video(name) and any(
            is_referenced(name, re.sub(standalone_pattern(name), "", t)) for t in texts
        ):
            report.warnings.append(
                f"{name}: cited mid-paragraph, so GitHub renders a link rather than a player"
            )

    report.warnings += [
        f"{name}: cited by nothing, so it is not uploaded"
        for name in sorted(by_name.keys() - cited)
    ]

    report.errors = list(dict.fromkeys(report.errors))
    report.warnings = list(dict.fromkeys(report.warnings))
    report.cited = sorted(cited)
    return report


def run_check(args: argparse.Namespace) -> None:
    if not args.target.is_file():
        print(f"ERROR: no target at {args.target}", file=sys.stderr)
        sys.exit(1)
    try:
        texts = target_bodies(args.target)
    except json.JSONDecodeError as e:
        print(f"ERROR: {args.target} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    report = check_media(args.dir, texts)
    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    if report.errors:
        print(f"ERROR: {args.target} cites media that will not render", file=sys.stderr)
        for error in report.errors:
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {len(report.cited)} media reference(s) resolve, {len(report.warnings)} warning(s)")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "upload-media",
        help="Upload media to GitHub user-attachments and point a review at the URLs",
    )
    parser.add_argument("--dir", type=Path, required=True, help="Directory holding the media")
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="File to rewrite: a .json pr-review payload, or any Markdown file",
    )
    parser.add_argument("--repository-id", help="Numeric repository id; required without --check")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report unresolvable references and exit non-zero, without uploading anything",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.check:
        run_check(args)
        return
    if not args.repository_id:
        print("--repository-id is required without --check", file=sys.stderr)
        sys.exit(2)
    if not args.target.is_file():
        print(f"No target at {args.target}; nothing to rewrite", file=sys.stderr)
        return
    if not args.dir.is_dir():
        print(f"No media directory at {args.dir}")
        return

    files, symlinks = collect_files(args.dir)
    for name in symlinks:
        print(f"  skip {name}: symlink", file=sys.stderr)
    if not files:
        print(f"No media in {args.dir}")
        return

    # Only what the review actually cites gets published. Captures taken to reason
    # with and then left uncited are scratch work.
    target_text = args.target.read_text()
    referenced = [p for p in files if is_referenced(p.name, target_text)]
    if unreferenced := [p.name for p in files if p not in referenced]:
        print(f"  not referenced, skipping: {', '.join(unreferenced)}", file=sys.stderr)
    if not referenced:
        print(f"No media referenced by {args.target}")
        return

    # A missing secret must still reach the rewrite below, or every reference ships
    # verbatim and renders as a broken repo-relative image.
    urls = {}
    if token := os.environ.get(TOKEN_ENV):
        print(f"Uploading {len(referenced)} referenced file(s) from {args.dir}")
        try:
            for path in referenced:
                if url := upload_asset(path, args.repository_id, token):
                    urls[path.name] = url
        except TokenRejected as e:
            # The step is continue-on-error, so without an annotation an expired
            # token would silently stop attaching media on every future review.
            print(f"::warning::{e}")
    else:
        print(f"{TOKEN_ENV} is unset; not uploading", file=sys.stderr)

    unavailable = [p.name for p in referenced if p.name not in urls]

    if args.target.suffix == ".json":
        payload = json.loads(target_text)
        rewritten = rewrite_payload(payload, urls, unavailable)
        args.target.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False))
    else:
        args.target.write_text(substitute(target_text, urls, unavailable))
    print(f"Embedded {len(urls)} of {len(referenced)} referenced file(s) into {args.target}")
