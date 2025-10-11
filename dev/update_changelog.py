import argparse
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import requests
from packaging.version import Version


def get_header_for_version(version):
    return "## {} ({})".format(version, datetime.now().strftime("%Y-%m-%d"))


def extract_pr_num_from_git_log_entry(git_log_entry):
    m = re.search(r"\(#(\d+)\)$", git_log_entry)
    return int(m.group(1)) if m else None


def format_label(label: str) -> str:
    key = label.split("/", 1)[-1]
    return {
        "model-registry": "Model Registry",
        "uiux": "UI",
    }.get(key, key.capitalize())


class PullRequest(NamedTuple):
    title: str
    number: int
    author: str
    labels: list[str]

    @property
    def url(self):
        return f"https://github.com/mlflow/mlflow/pull/{self.number}"

    @property
    def release_note_labels(self):
        return [l for l in self.labels if l.startswith("rn/")]

    def __str__(self):
        areas = " / ".join(
            sorted(
                map(
                    format_label,
                    filter(lambda l: l.split("/")[0] in ("area", "language"), self.labels),
                )
            )
        )
        return f"[{areas}] {self.title} (#{self.number}, @{self.author})"

    def __repr__(self):
        return str(self)


class Section(NamedTuple):
    title: str
    items: list[Any]

    def __str__(self):
        if not self.items:
            return ""
        return "\n\n".join(
            [
                self.title,
                "\n".join(f"- {item}" for item in self.items),
            ]
        )


def is_shallow():
    return (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--is-shallow-repository",
            ],
            text=True,
        ).strip()
        == "true"
    )


def batch_fetch_prs_graphql(pr_numbers: list[int]) -> list[PullRequest]:
    """
    Batch fetch PR data using GitHub GraphQL API.
    """
    if not pr_numbers:
        return []

    # GitHub GraphQL has query size limits, so batch in chunks
    MAX_PRS_PER_QUERY = 50  # Conservative limit to avoid query size issues
    all_prs: list[PullRequest] = []

    for i in range(0, len(pr_numbers), MAX_PRS_PER_QUERY):
        chunk = pr_numbers[i : i + MAX_PRS_PER_QUERY]
        chunk_prs = _fetch_pr_chunk_graphql(chunk)
        all_prs.extend(chunk_prs)

    return all_prs


def _fetch_pr_chunk_graphql(pr_numbers: list[int]) -> list[PullRequest]:
    """
    Fetch a chunk of PRs using GraphQL.
    """
    # Build GraphQL query with aliases for each PR
    query_parts = [
        "query($owner: String!, $repo: String!) {",
        "  repository(owner: $owner, name: $repo) {",
    ]

    for i, pr_num in enumerate(pr_numbers):
        query_parts.append(f"""
    pr{i}: pullRequest(number: {pr_num}) {{
      number
      title
      author {{
        login
      }}
      labels(first: 100) {{
        nodes {{
          name
        }}
      }}
    }}""")

    query_parts.extend(["  }", "}"])
    query = "\n".join(query_parts)

    # Headers with authentication
    headers = {"Content-Type": "application/json"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    print(f"Batch fetching {len(pr_numbers)} PRs with GraphQL...")
    resp = requests.post(
        "https://api.github.com/graphql",
        json={
            "query": query,
            "variables": {"owner": "mlflow", "repo": "mlflow"},
        },
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")

    # Extract PR data from response and create PullRequest objects
    repository_data = data["data"]["repository"]
    prs = []
    for i, pr_num in enumerate(pr_numbers):
        pr_info = repository_data.get(f"pr{i}")
        if pr_info and pr_info.get("author"):
            prs.append(
                PullRequest(
                    title=pr_info["title"],
                    number=pr_info["number"],
                    author=pr_info["author"]["login"],
                    labels=[label["name"] for label in pr_info["labels"]["nodes"]],
                )
            )
        else:
            print(f"Warning: Could not fetch data for PR #{pr_num}")

    return prs


def main(prev_version, release_version, remote):
    if is_shallow():
        print("Unshallowing repository to ensure `git log` works correctly")
        subprocess.check_call(["git", "fetch", "--unshallow"])
        print("Modifying .git/config to fetch remote branches")
        subprocess.check_call(
            ["git", "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"]
        )
    release_tag = f"v{prev_version}"
    ver = Version(release_version)
    branch = f"branch-{ver.major}.{ver.minor}"
    subprocess.check_call(["git", "fetch", remote, "tag", release_tag])
    subprocess.check_call(["git", "fetch", remote, branch])
    git_log_output = subprocess.check_output(
        [
            "git",
            "log",
            "--left-right",
            "--graph",
            "--cherry-pick",
            "--pretty=format:%s",
            f"tags/{release_tag}...{remote}/{branch}",
        ],
        text=True,
    )
    logs = [l[2:] for l in git_log_output.splitlines() if l.startswith("> ")]

    # Extract all PR numbers first
    pr_numbers = []
    for log in logs:
        if pr_num := extract_pr_num_from_git_log_entry(log):
            pr_numbers.append(pr_num)

    prs = batch_fetch_prs_graphql(pr_numbers)
    label_to_prs = defaultdict(list)
    author_to_prs = defaultdict(list)
    unlabelled_prs = []
    for pr in prs:
        if pr.author == "mlflow-app":
            continue

        if len(pr.release_note_labels) == 0:
            unlabelled_prs.append(pr)

        for label in pr.release_note_labels:
            if label == "rn/none":
                author_to_prs[pr.author].append(pr)
            else:
                label_to_prs[label].append(pr)

    assert len(unlabelled_prs) == 0, "The following PRs need to be categorized:\n" + "\n".join(
        f"- {pr.url}" for pr in unlabelled_prs
    )

    unknown_labels = set(label_to_prs.keys()) - {
        "rn/highlight",
        "rn/feature",
        "rn/breaking-change",
        "rn/bug-fix",
        "rn/documentation",
        "rn/none",
    }
    assert len(unknown_labels) == 0, f"Unknown labels: {unknown_labels}"

    breaking_changes = Section("Breaking changes:", label_to_prs.get("rn/breaking-change", []))
    highlights = Section("Major new features:", label_to_prs.get("rn/highlight", []))
    features = Section("Features:", label_to_prs.get("rn/feature", []))
    bug_fixes = Section("Bug fixes:", label_to_prs.get("rn/bug-fix", []))
    doc_updates = Section("Documentation updates:", label_to_prs.get("rn/documentation", []))
    small_updates = [
        ", ".join([f"#{pr.number}" for pr in prs] + [f"@{author}"])
        for author, prs in author_to_prs.items()
    ]
    small_updates = "Small bug fixes and documentation updates:\n\n" + "; ".join(small_updates)
    sections = filter(
        str.strip,
        map(
            str,
            [
                get_header_for_version(release_version),
                f"MLflow {release_version} includes several major features and improvements",
                breaking_changes,
                highlights,
                features,
                bug_fixes,
                doc_updates,
                small_updates,
            ],
        ),
    )
    new_changelog = "\n\n".join(sections)
    changelog_header = "# CHANGELOG"
    changelog = Path("CHANGELOG.md")
    old_changelog = changelog.read_text().replace(f"{changelog_header}\n\n", "", 1)
    new_changelog = "\n\n".join(
        [
            changelog_header,
            new_changelog,
            old_changelog,
        ]
    )
    changelog.write_text(new_changelog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update CHANGELOG.md")
    parser.add_argument("--prev-version", required=True, help="Previous version")
    parser.add_argument("--release-version", required=True, help="MLflow version to release")
    parser.add_argument("--remote", default="origin", help="Git remote to use (default: origin)")
    args = parser.parse_args()
    main(args.prev_version, args.release_version, args.remote)
