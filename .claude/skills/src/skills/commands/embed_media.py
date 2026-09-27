# ruff: noqa: T201
"""Upload review media to GitHub's user-attachments store and swap filenames for URLs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skills.github.uploads import (
    MIME_TYPES,
    UploadFailed,
    is_video,
    max_bytes,
    upload_asset,
)
from skills.github.utils import resolve_github_token


def link_target(cited: str) -> str:
    return rf"\]\({re.escape(cited)}\)"


def is_referenced(cited: str, text: str) -> bool:
    return re.search(link_target(cited), text) is not None


def standalone_pattern(cited: str) -> str:
    # GitHub renders a video player only for a bare URL alone in its own paragraph.
    return rf"(?m)^[ \t]*!?\[[^\]]*{link_target(cited)}[ \t]*$"


def substitute(text: str, urls: dict[str, str], unavailable: Iterable[str] = ()) -> str:
    for cited, url in urls.items():
        link = link_target(cited)
        if is_video(cited):
            # Promote a reference that already sits on its own line.
            text = re.sub(standalone_pattern(cited), f"\n{url}\n", text)
            # Whatever is left is mid-sentence, where ![]() around a video URL renders
            # as a broken image. Drop the bang so it degrades to a link instead.
            text = re.sub(rf"!(?=\[[^\]]*{link})", "", text)
        text = re.sub(link, f"]({url})", text)

    # A citation with no URL (upload failed, or the file was skipped) would otherwise
    # post the local path, which resolves to nothing. Strip the markup so it degrades
    # to prose.
    for cited in unavailable:
        link = link_target(cited)
        text = re.sub(rf"!?\[{link}", Path(cited).name, text)
        text = re.sub(rf"!?\[([^\]]+){link}", r"\1", text)
    return text


def collect_files(directory: Path) -> tuple[list[Path], list[str]]:
    """Return the uploadable files, and the names rejected for being symlinks.

    ``is_file()`` follows symlinks, so a link planted here (say secret.png ->
    /proc/self/environ) would publish this process's own GH_TOKEN as an
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

    cited: set[str] = set()
    for text in texts:
        for raw in LINK.findall(text):
            name = raw.rsplit("/", 1)[-1]
            path = by_name.get(name)
            if path and raw == str(path):
                cited.add(name)
            # A basename alone can only be meant as a capture; the same basename under
            # some other directory is an ordinary link that happens to collide.
            elif path and "/" not in raw.removeprefix("./"):
                cited.add(name)
                report.errors.append(f"({raw}): cite the capture as {path}")
            elif raw.startswith(f"{directory}/"):
                report.errors.append(
                    f"({raw}): no such file, so the citation is stripped and the finding "
                    "degrades to prose"
                )

    for name in sorted(cited):
        path = by_name[name]
        cite = str(path)
        size = path.stat().st_size
        if Path(name).suffix.lower() not in MIME_TYPES:
            report.errors.append(f"{name}: unsupported extension, so the reference is dropped")
        elif size == 0:
            report.errors.append(f"{name}: empty, so the reference is dropped")
        elif size > (limit := max_bytes(name)):
            report.errors.append(
                f"{name}: {size} bytes exceeds the {limit} byte cap, so the reference is dropped"
            )
        elif is_video(name) and any(
            is_referenced(cite, re.sub(standalone_pattern(cite), "", t)) for t in texts
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
        "embed-media",
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
    files, symlinks = collect_files(args.dir) if args.dir.is_dir() else ([], [])
    for name in symlinks:
        print(f"  skip {name}: symlink", file=sys.stderr)

    # Only what the review actually cites gets published. Captures taken to reason
    # with and then left uncited are scratch work.
    target_text = args.target.read_text()
    referenced = [p for p in files if is_referenced(str(p), target_text)]
    if unreferenced := [p.name for p in files if p not in referenced]:
        print(f"  not referenced, skipping: {', '.join(unreferenced)}", file=sys.stderr)

    # A citation naming no capture resolves to nothing, so it has to be stripped rather
    # than posted. --check reports it first, but Claude runs that; this step is the last
    # thing between a typo and the review.
    resolvable = {str(p) for p in referenced}
    names = {p.name for p in files}
    stray = [
        raw
        for raw in dict.fromkeys(LINK.findall(target_text))
        if raw not in resolvable
        and (raw.startswith(f"{args.dir}/") or raw.removeprefix("./") in names)
    ]
    for raw in stray:
        print(f"  no such capture, stripping: {raw}", file=sys.stderr)
    if not referenced and not stray:
        print(f"No media referenced by {args.target}")
        return

    # A missing credential must still reach the rewrite below, or every reference ships
    # verbatim and posts a local path.
    urls = {}
    token = resolve_github_token()
    if not token:
        print("no GitHub token; not uploading", file=sys.stderr)
    elif referenced:
        print(f"Uploading {len(referenced)} referenced file(s) from {args.dir}")
        for path in referenced:
            try:
                urls[str(path)] = upload_asset(path, args.repository_id, token)
            except UploadFailed as e:
                # A fault that is not about this file fails every remaining upload, so
                # stop rather than retry. The step is continue-on-error, so without an
                # annotation this would silently stop attaching media on every future
                # review.
                if e.fatal:
                    print(f"::warning::media upload stopped: {e}")
                    break
                print(f"  failed {e}", file=sys.stderr)
            else:
                print(f"  uploaded {path.name} -> {urls[str(path)]}")

    unavailable = [str(p) for p in referenced if str(p) not in urls] + stray

    if args.target.suffix == ".json":
        payload = json.loads(target_text)
        rewritten = rewrite_payload(payload, urls, unavailable)
        args.target.write_text(json.dumps(rewritten, indent=2, ensure_ascii=False))
    else:
        args.target.write_text(substitute(target_text, urls, unavailable))
    print(f"Embedded {len(urls)} of {len(referenced)} referenced file(s) into {args.target}")
