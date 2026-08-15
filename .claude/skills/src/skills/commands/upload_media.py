# ruff: noqa: T201
"""Upload review media to GitHub's user-attachments store and swap filenames for URLs."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
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

# 10MB covers images and video on free plans; the smaller bound is the safe one.
MAX_BYTES = 10 * 1024 * 1024


def is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def upload_asset(
    path: Path,
    repository_id: str,
    token: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
    max_bytes: int = MAX_BYTES,
) -> str | None:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        print(f"  skip {path.name}: unsupported extension", file=sys.stderr)
        return None

    size = path.stat().st_size
    if size > max_bytes:
        print(f"  skip {path.name}: {size} bytes exceeds {max_bytes}", file=sys.stderr)
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
        with opener(request, timeout=60) as resp:
            body = json.load(resp)
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


def substitute(text: str, urls: dict[str, str], unavailable: Iterable[str] = ()) -> str:
    for name, url in urls.items():
        quoted = re.escape(name)
        if is_video(name):
            # GitHub renders a player only for a bare URL alone in its own paragraph,
            # so promote a reference that already sits on its own line.
            standalone = rf"(?m)^[ \t]*(?:!?\[[^\]]*\]\((?:\./)?{quoted}\)|`{quoted}`)[ \t]*$"
            text = re.sub(standalone, f"\n{url}\n", text)
            # Whatever is left is mid-sentence, where ![]() around a video URL renders
            # as a broken image. Drop the bang so it degrades to a link instead.
            text = re.sub(rf"!(?=\[[^\]]*\]\((?:\./)?{quoted}\))", "", text)
        text = re.sub(rf"\]\((?:\./)?{quoted}\)", f"]({url})", text)
        # Lookarounds keep a second pass a no-op.
        text = re.sub(rf"(?<!\[)`{quoted}`(?!\])", f"[`{name}`]({url})", text)

    # A name with no URL (upload failed, or the file was skipped) would otherwise
    # survive as ![desc](shot.png), which GitHub resolves repo-relative and renders
    # as a broken image. Strip the markup so it degrades to prose.
    for name in unavailable:
        quoted = re.escape(name)
        text = re.sub(rf"!?\[\]\((?:\./)?{quoted}\)", name, text)
        text = re.sub(rf"!?\[([^\]]+)\]\((?:\./)?{quoted}\)", r"\1", text)
    return text


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
    parser.add_argument("--repository-id", required=True, help="Numeric repository id")
    parser.set_defaults(func=run)


def run(
    args: argparse.Namespace,
    upload: Callable[..., str | None] = upload_asset,
) -> None:
    if not args.target.is_file():
        print(f"No target at {args.target}; nothing to rewrite", file=sys.stderr)
        return
    if not args.dir.is_dir():
        print(f"No media directory at {args.dir}")
        return

    # is_file() follows symlinks, so a link planted here (say secret.png ->
    # /proc/self/environ) would publish this process's own UPLOAD_MEDIA_TOKEN as an
    # attachment. Claude writes this directory, and a poisoned diff steers Claude.
    files: list[Path] = []
    for path in sorted(args.dir.iterdir()):
        if path.is_symlink():
            print(f"  skip {path.name}: symlink", file=sys.stderr)
        elif path.is_file():
            files.append(path)
    if not files:
        print(f"No media in {args.dir}")
        return

    # A missing secret must still reach the rewrite below, or every reference ships
    # verbatim and renders as a broken repo-relative image.
    urls = {}
    if token := os.environ.get(TOKEN_ENV):
        print(f"Uploading {len(files)} file(s) from {args.dir}")
        for path in files:
            if url := upload(path, args.repository_id, token):
                urls[path.name] = url
    else:
        print(f"{TOKEN_ENV} is unset; not uploading", file=sys.stderr)

    unavailable = [p.name for p in files if p.name not in urls]

    if args.target.suffix == ".json":
        payload = json.loads(args.target.read_text())
        rewritten = rewrite_payload(payload, urls, unavailable)
        args.target.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False))
    else:
        args.target.write_text(substitute(args.target.read_text(), urls, unavailable))
    print(f"Embedded {len(urls)} of {len(files)} file(s) into {args.target}")
