"""Build MLflow release documentation and publish to mlflow-legacy-website."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import InvalidVersion, Version

# Local git operations finish in seconds, even staging a full docs build. Pushing
# one is the outlier, since it uploads thousands of files.
_GIT_TIMEOUT = 5 * 60
_GIT_PUSH_TIMEOUT = 30 * 60


class Repo:
    """A website repository checkout that this script commits and pushes to.

    The caller provides the checkout, its credentials and the committer identity;
    `gh` picks up `$GH_TOKEN` from the environment.
    """

    def __init__(self, repo: str, root: Path, *, default_branch: str = "main"):
        self.repo = repo
        self.root = root
        self.default_branch = default_branch

    def git(self, *args: str, timeout: int = _GIT_TIMEOUT) -> None:
        subprocess.check_call(["git", *args], cwd=self.root, timeout=timeout)

    def has_changes(self) -> bool:
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root)
        return result.returncode != 0

    def create_pr(self, *, head: str, title: str, body: str) -> str:
        output = subprocess.check_output(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                self.repo,
                "--head",
                head,
                "--base",
                self.default_branch,
                "--title",
                title,
                "--body",
                body,
            ],
            text=True,
        )
        return output.strip()


def _read_version(repo_root: Path) -> str:
    output = subprocess.check_output(["uv", "version", "--short"], cwd=repo_root, text=True)
    return output.strip()


def build_docs(args: argparse.Namespace) -> None:
    mlflow_dir = Path(args.mlflow_dir).resolve()
    release_version = _read_version(mlflow_dir)
    print(f"Building docs for MLflow {release_version}")

    subprocess.check_call(["uv", "sync", "--group", "docs", "--extra", "gateway"], cwd=mlflow_dir)
    subprocess.check_call(["uv", "pip", "install", "-r", "requirements/torch.txt"], cwd=mlflow_dir)
    docs_dir = mlflow_dir / "docs"
    env = {**os.environ, "GTM_ID": args.gtm_id}
    subprocess.check_call(["npm", "ci"], cwd=docs_dir, env=env)
    subprocess.check_call(["npm", "run", "build-all", "--", "--use-npm"], cwd=docs_dir, env=env)

    website_repo = Repo("mlflow/mlflow-legacy-website", Path(args.website_dir).resolve())
    branch_name = f"docs-{release_version}-{uuid.uuid4().hex[:8]}"
    website_repo.git("checkout", "-q", "-b", branch_name)

    version = Version(release_version)

    _remove_stale_prereleases(website_repo.root / "docs", version)

    # Copy built docs. `build-all.py` produces a separate build per target
    # (e.g. `build/3.11.1` and `build/latest`), each with its own baseUrl,
    # so we must copy from the matching directory rather than reusing one source.
    for dest_name in _version_targets(version, website_repo.root):
        src = docs_dir / "build" / dest_name
        dst = website_repo.root / "docs" / dest_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # Update versions.json
    _update_versions_json(website_repo.root / "docs" / "versions.json", version)

    website_repo.git("add", "-A")

    if not website_repo.has_changes():
        print("No changes to commit, skipping.")
        return

    # `-q`: a docs publish creates ~6k files, and git lists every one of them
    website_repo.git("commit", "-q", "-m", "Add docs")

    if args.dry_run:
        return

    website_repo.git("push", "origin", branch_name, timeout=_GIT_PUSH_TIMEOUT)
    pr_url = website_repo.create_pr(
        head=branch_name,
        title=f"Add documentation for {release_version}",
        body="",
    )
    print(f"Created {pr_url}")


def _remove_stale_prereleases(docs_root: Path, version: Version) -> None:
    """Drop release candidate docs for `version` once the final release is published."""
    if version.is_prerelease:
        return

    for p in docs_root.iterdir():
        if not p.is_dir():
            continue
        try:
            v = Version(p.name)
        except InvalidVersion:
            continue
        if v.is_prerelease and v.base_version == version.base_version:
            shutil.rmtree(p)


def _version_targets(version: Version, website_root: Path) -> list[str]:
    json_path = website_root / "docs" / "versions.json"
    versions_json = json.loads(json_path.read_text())
    latest_version = max(map(Version, versions_json["versions"]))
    targets = [str(version)]
    if version >= latest_version:
        targets.append("latest")
    return targets


def _update_versions_json(json_path: Path, release_version: Version) -> None:
    data = json.loads(json_path.read_text())
    versions = [Version(v) for v in data["versions"]]
    if release_version not in versions:
        versions.append(release_version)

    # Keep only the highest version for each minor release
    latest_by_minor: dict[tuple[int, int], Version] = {}
    for v in versions:
        key = (v.major, v.minor)
        if key not in latest_by_minor or v > latest_by_minor[key]:
            latest_by_minor[key] = v

    data["versions"] = [str(v) for v in sorted(latest_by_minor.values(), reverse=True)]
    json_path.write_text(json.dumps(data, indent=2))


def release_post(args: argparse.Namespace) -> None:
    mlflow_dir = Path(args.mlflow_dir).resolve()
    release_version = _read_version(mlflow_dir)
    print(f"Creating release post for MLflow {release_version}")

    website_repo = Repo("mlflow/mlflow-website", Path(args.website_dir).resolve())
    branch_name = f"release-post-{release_version}-{uuid.uuid4().hex[:8]}"
    website_repo.git("checkout", "-q", "-b", branch_name)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    name = f"{today}-{release_version}-release.md"
    post_path = website_repo.root / "website" / "releases" / name
    post_path.write_text(_render_release_post(Version(release_version)))

    website_repo.git("add", "-A")

    if not website_repo.has_changes():
        print("No changes to commit, skipping.")
        return

    website_repo.git("commit", "-q", "-m", "Add release post")

    if args.dry_run:
        return

    website_repo.git("push", "origin", branch_name, timeout=_GIT_PUSH_TIMEOUT)
    pr_url = website_repo.create_pr(
        head=branch_name,
        title=f"Add release post for {release_version}",
        body="Be sure to fill in the contents",
    )
    print(f"Created {pr_url}")


def _render_release_post(version: Version) -> str:
    if version.is_prerelease:
        return _RC_TEMPLATE.format(version=version, base_version=version.base_version)
    return _RELEASE_TEMPLATE.format(version=version)


_RELEASE_TEMPLATE = """\
---
title: MLflow {version}
slug: {version}
authors: [mlflow-maintainers]
---

<REPLACE_ME>

For a comprehensive list of changes, see the
[release change log](https://github.com/mlflow/mlflow/releases/tag/v{version}),
and check out the latest documentation on [mlflow.org](http://mlflow.org/).
"""

_RC_TEMPLATE = """\
---
title: MLflow {version}
slug: {version}
authors: [mlflow-maintainers]
---

MLflow {version} is a release candidate for {base_version}. To install, run the following command:

```sh
pip install mlflow=={version}
```

<!-- Major changes that need to be highlighted in the release post go here -->
<REPLACE_ME>

Please try it out and report any issues on [the issue tracker](https://github.com/mlflow/mlflow/issues).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLflow release documentation tools")
    parser.add_argument(
        "--mlflow-dir",
        default=".",
        help="Path to the local MLflow repository checkout",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip push and PR creation",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    docs_parser = subparsers.add_parser("build-docs", help="Build and publish MLflow documentation")
    docs_parser.add_argument("--gtm-id", default="GTM-TEST", help="Google Tag Manager ID")
    docs_parser.add_argument(
        "--website-dir",
        required=True,
        help="Path to a mlflow/mlflow-legacy-website checkout to publish into",
    )
    post_parser = subparsers.add_parser("release-post", help="Create a release post")
    post_parser.add_argument(
        "--website-dir",
        required=True,
        help="Path to a mlflow/mlflow-website checkout to publish into",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match args.command:
        case "build-docs":
            build_docs(args)
        case "release-post":
            release_post(args)
        case _:
            raise ValueError(f"Unexpected command: {args.command!r}")


if __name__ == "__main__":
    main()
