# ruff: noqa: T201
"""Upload review media to GitHub's user-attachments store and swap filenames for URLs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

UPLOAD_URL = "https://uploads.github.com/user-attachments/assets"
# Read from the environment, never argv: a PAT in a CLI argument is visible in the
# process list for the life of the call.
TOKEN_ENV = "MEDIA_TOKEN"

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

# 10MB covers images and video on free plans; the smaller bound is the safe one.
MAX_BYTES = 10 * 1024 * 1024


def upload_asset(path: Path, repository_id: str, token: str) -> str | None:
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        print(f"  skip {path.name}: unsupported extension", file=sys.stderr)
        return None

    size = path.stat().st_size
    if size > MAX_BYTES:
        print(f"  skip {path.name}: {size} bytes exceeds {MAX_BYTES}", file=sys.stderr)
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
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"  failed {path.name}: {e}", file=sys.stderr)
        return None

    match body:
        case {"url": str(url)} if url:
            print(f"  uploaded {path.name} -> {url}")
            return url
        case _:
            print(f"  failed {path.name}: response carried no url", file=sys.stderr)
            return None


def substitute(text: str, urls: dict[str, str]) -> str:
    for name, url in urls.items():
        quoted = re.escape(name)
        text = re.sub(rf"\]\((?:\./)?{quoted}\)", f"]({url})", text)
        # Lookarounds keep a second pass a no-op.
        text = re.sub(rf"(?<!\[)`{quoted}`(?!\])", f"[`{name}`]({url})", text)
    return text


def rewrite_payload(payload: dict[str, Any], urls: dict[str, str]) -> dict[str, Any]:
    # The schema pins body to end with the Claude footer, so only substitute, never append.
    match payload:
        case {"body": str(body)}:
            payload["body"] = substitute(body, urls)
    match payload:
        case {"comments": [*comments]}:
            for comment in comments:
                match comment:
                    case {"body": str(body)}:
                        comment["body"] = substitute(body, urls)
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


def run(args: argparse.Namespace) -> None:
    if not (token := os.environ.get(TOKEN_ENV)):
        print(f"{TOKEN_ENV} is unset; skipping media upload", file=sys.stderr)
        return
    if not args.target.is_file():
        print(f"No target at {args.target}; nothing to rewrite", file=sys.stderr)
        return
    if not args.dir.is_dir():
        print(f"No media directory at {args.dir}")
        return

    files = sorted(p for p in args.dir.iterdir() if p.is_file())
    if not files:
        print(f"No media in {args.dir}")
        return

    print(f"Uploading {len(files)} file(s) from {args.dir}")
    urls = {}
    for path in files:
        if url := upload_asset(path, args.repository_id, token):
            urls[path.name] = url

    if not urls:
        print("No uploads succeeded; leaving the target unchanged", file=sys.stderr)
        return

    if args.target.suffix == ".json":
        payload = json.loads(args.target.read_text())
        rewritten = rewrite_payload(payload, urls)
        args.target.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False))
    else:
        args.target.write_text(substitute(args.target.read_text(), urls))
    print(f"Embedded {len(urls)} of {len(files)} file(s) into {args.target}")
