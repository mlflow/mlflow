# ruff: noqa: T201
"""List GitHub repository security advisories, grouped by state."""

from __future__ import annotations

import argparse
import asyncio
import sys

from skills.github import GitHubClient, SecurityAdvisory

STATES = ["triage", "draft", "published", "closed"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_repo(repo: str) -> tuple[str, str]:
    owner, _, name = repo.partition("/")
    if not owner or not name:
        log(f"Error: Invalid repo '{repo}'. Expected 'owner/repo'.")
        sys.exit(1)
    return owner, name


def format_advisory(advisory: SecurityAdvisory) -> str:
    severity = advisory.severity or "unknown"
    cwes = ", ".join(advisory.cwe_ids) if advisory.cwe_ids else "no CWE"
    return f"- {advisory.ghsa_id} ({severity}, {cwes})\n  {advisory.html_url}\n  {advisory.summary}"


def format_output(advisories: list[SecurityAdvisory]) -> str:
    by_state: dict[str, list[SecurityAdvisory]] = {state: [] for state in STATES}
    for advisory in advisories:
        by_state.setdefault(advisory.state, []).append(advisory)

    counts = " | ".join(f"{len(by_state.get(state, []))} {state}" for state in STATES)
    sections = [f"# Security advisories ({len(advisories)} total)", "", counts]

    for state in STATES:
        items = by_state.get(state, [])
        sections.append("")
        sections.append(f"## {state} ({len(items)})")
        if items:
            sections.extend(format_advisory(a) for a in items)
        else:
            sections.append("(none)")
    return "\n".join(sections)


async def list_advisories(repo: str, state: str | None) -> str:
    owner, name = parse_repo(repo)
    log(f"Fetching security advisories for {owner}/{name}" + (f" (state={state})" if state else ""))
    async with GitHubClient() as client:
        advisories = [a async for a in client.get_security_advisories(owner, name, state)]
    log(f"Found {len(advisories)} advisory(ies)")
    return format_output(advisories)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("list-advisories", help="List repository security advisories")
    parser.add_argument(
        "--repo", default="mlflow/mlflow", help="owner/repo (default: mlflow/mlflow)"
    )
    parser.add_argument("--state", choices=STATES, help="Filter by advisory state (default: all)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    print(asyncio.run(list_advisories(args.repo, args.state)))
