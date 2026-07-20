# ruff: noqa: T201
"""Fetch full detail for one or more security advisories by GHSA id (read-only)."""

from __future__ import annotations

import argparse
import asyncio
import sys

from skills.github import GitHubClient, SecurityAdvisory, Vulnerability


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_repo(repo: str) -> tuple[str, str]:
    owner, _, name = repo.partition("/")
    if not owner or not name:
        log(f"Error: Invalid repo '{repo}'. Expected 'owner/repo'.")
        sys.exit(1)
    return owner, name


def format_vulnerability(vuln: Vulnerability) -> str:
    package = "unknown package"
    if vuln.package:
        ecosystem = vuln.package.ecosystem or "?"
        name = vuln.package.name or "?"
        package = f"{name} ({ecosystem})"
    parts = [
        f"  - package: {package}",
        f"    affected: {vuln.vulnerable_version_range or 'unspecified'}",
        f"    patched: {vuln.patched_versions or 'none'}",
    ]
    if vuln.vulnerable_functions:
        parts.append(f"    functions: {', '.join(vuln.vulnerable_functions)}")
    return "\n".join(parts)


def format_advisory(advisory: SecurityAdvisory) -> str:
    reporter = advisory.author.login if advisory.author and advisory.author.login else "unknown"
    cwes = ", ".join(advisory.cwe_ids) if advisory.cwe_ids else "no CWE"
    lines = [
        f"# {advisory.ghsa_id}",
        f"summary: {advisory.summary}",
        f"state: {advisory.state}",
        f"severity: {advisory.severity or 'unknown'}",
        f"cwe: {cwes}",
        f"cve: {advisory.cve_id or 'no CVE'}",
        f"reporter: {reporter}",
        f"created: {advisory.created_at or 'unknown'}",
        f"updated: {advisory.updated_at or 'unknown'}",
        f"published: {advisory.published_at or 'not published'}",
        f"url: {advisory.html_url}",
    ]

    lines.append("vulnerabilities:")
    if advisory.vulnerabilities:
        lines.extend(format_vulnerability(v) for v in advisory.vulnerabilities)
    else:
        lines.append("  (none listed)")

    lines.append("")
    lines.append("## Description")
    lines.append(advisory.description or "(no description provided)")
    return "\n".join(lines)


async def get_advisories(repo: str, ghsa_ids: list[str]) -> str:
    owner, name = parse_repo(repo)
    log(f"Fetching {len(ghsa_ids)} advisory(ies) for {owner}/{name}")
    async with GitHubClient() as client:
        # gather preserves input order so multi-id output is deterministic.
        advisories = await asyncio.gather(
            *(client.get_security_advisory(owner, name, ghsa_id) for ghsa_id in ghsa_ids)
        )
    return "\n\n---\n\n".join(format_advisory(a) for a in advisories)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "get-advisory", help="Fetch full detail for one or more security advisories"
    )
    parser.add_argument("ghsa_ids", nargs="+", help="One or more GHSA identifiers")
    parser.add_argument(
        "--repo", default="mlflow/mlflow", help="owner/repo (default: mlflow/mlflow)"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    print(asyncio.run(get_advisories(args.repo, args.ghsa_ids)))
