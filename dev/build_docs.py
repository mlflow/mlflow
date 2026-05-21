"""Build MLflow release documentation and publish to mlflow-legacy-website."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import InvalidVersion, Version


class Repo:
    def __init__(self, repo: str, root: Path, *, default_branch: str, token: str | None = None):
        self.repo = repo
        self.root = root
        self.default_branch = default_branch
        self.token = token

    @classmethod
    @contextmanager
    def clone(
        cls,
        *,
        repo: str,
        branch: str,
        token: str | None = None,
    ) -> Iterator[Repo]:
        if token:
            url = f"https://mlflow-app[bot]:{token}@github.com/{repo}.git"
        else:
            url = f"https://github.com/{repo}.git"
        cmd = ["git", "clone", "--depth", "1", "--branch", branch]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            subprocess.check_call([*cmd, url, root])
            instance = cls(repo, root, default_branch=branch, token=token)
            instance._configure_identity()
            yield instance

    def _configure_identity(self) -> None:
        self.git("config", "user.name", "mlflow-app[bot]")
        self.git("config", "user.email", "mlflow-app[bot]@users.noreply.github.com")

    @property
    def branch(self) -> str:
        output = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=self.root, text=True
        )
        return output.strip()

    def git(self, *args: str) -> None:
        subprocess.check_call(["git", *args], cwd=self.root)

    def checkout_new(self, branch: str) -> None:
        self.git("checkout", "-b", branch)

    def add_all(self) -> None:
        self.git("add", "-A")

    def has_changes(self) -> bool:
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root)
        return result.returncode != 0

    def commit(self, message: str) -> None:
        self.git("commit", "-m", message)

    def push(self) -> None:
        self.git("push", "origin", self.branch)

    def create_pr(self, *, title: str, body: str) -> str:
        if not self.token:
            raise ValueError("Cannot create PR without a token")
        env = {**os.environ, "GH_TOKEN": self.token}
        output = subprocess.check_output(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                self.repo,
                "--head",
                self.branch,
                "--base",
                self.default_branch,
                "--title",
                title,
                "--body",
                body,
            ],
            text=True,
            env=env,
        )
        return output.strip()


def _read_version(repo_root: Path) -> str:
    # `uv version` outputs "<name> <version>", e.g. "mlflow 3.11.0"
    output = subprocess.check_output(["uv", "version"], cwd=repo_root, text=True)
    return output.strip().split()[-1]


def build_docs(args: argparse.Namespace) -> None:
    mlflow_dir = Path(args.mlflow_dir).resolve()
    release_version = _read_version(mlflow_dir)
    print(f"Building docs for MLflow {release_version}")

    subprocess.check_call(["uv", "sync", "--group", "docs", "--extra", "gateway"], cwd=mlflow_dir)
    subprocess.check_call(
        ["uv", "pip", "install", "-r", "requirements/torch.txt"], cwd=mlflow_dir
    )
    docs_dir = mlflow_dir / "docs"
    env = {**os.environ, "GTM_ID": args.gtm_id}
    subprocess.check_call(["npm", "ci"], cwd=docs_dir, env=env)
    subprocess.check_call(["npm", "run", "build-all", "--", "--use-npm"], cwd=docs_dir, env=env)

    with Repo.clone(
        repo="mlflow/mlflow-legacy-website",
        branch="main",
        token=args.token,
    ) as website_repo:
        branch_name = f"docs-{release_version}-{uuid.uuid4().hex[:8]}"
        website_repo.checkout_new(branch_name)

        version = Version(release_version)

        # Clean up release candidate docs when publishing a final release
        if not version.is_prerelease:
            for p in (website_repo.root / "docs").iterdir():
                if not p.is_dir():
                    continue
                try:
                    v = Version(p.name)
                except InvalidVersion:
                    continue
                if v.is_prerelease and v.base_version == version.base_version:
                    shutil.rmtree(p)

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

        website_repo.add_all()

        if not website_repo.has_changes():
            print("No changes to commit, skipping.")
            return

        website_repo.commit("Add docs")

        if args.dry_run:
            return

        website_repo.push()

        if args.token:
            pr_url = website_repo.create_pr(
                title=f"Add documentation for {release_version}",
                body="",
            )
            print(f"Created {pr_url}")


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

    with Repo.clone(
        repo="mlflow/mlflow-website",
        branch="main",
        token=args.token,
    ) as website_repo:
        branch_name = f"release-post-{release_version}-{uuid.uuid4().hex[:8]}"
        website_repo.checkout_new(branch_name)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        name = f"{today}-{release_version}-release.md"
        post_path = website_repo.root / "website" / "releases" / name

        if "rc" in release_version:
            base_version = release_version.split("rc")[0]
            content = _RC_TEMPLATE.format(version=release_version, base_version=base_version)
        else:
            content = _RELEASE_TEMPLATE.format(version=release_version)

        post_path.write_text(content)

        website_repo.add_all()

        if not website_repo.has_changes():
            print("No changes to commit, skipping.")
            return

        website_repo.commit("Add release post")

        if args.dry_run:
            return

        website_repo.push()

        if args.token:
            pr_url = website_repo.create_pr(
                title=f"Add release post for {release_version}",
                body="Be sure to fill in the contents",
            )
            print(f"Created {pr_url}")


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
        "--token",
        default=os.environ.get("GH_TOKEN"),
        help="GitHub token for pushing and creating PRs (default: $GH_TOKEN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip push and PR creation",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    docs_parser = subparsers.add_parser("build-docs", help="Build and publish MLflow documentation")
    docs_parser.add_argument("--gtm-id", default="GTM-TEST", help="Google Tag Manager ID")
    subparsers.add_parser("release-post", help="Create a release post")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match args.command:
        case "build-docs":
            build_docs(args)
        case "release-post":
            release_post(args)


if __name__ == "__main__":
    main()
