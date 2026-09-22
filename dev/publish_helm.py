"""Package and publish a stable MLflow release's Helm chart."""

import argparse
import json
import os
import re
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import yaml

CHART_REPOSITORY = "mlflow/charts/mlflow"
CHART_URI = f"oci://ghcr.io/{CHART_REPOSITORY}"


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def release_version(tag: str) -> str:
    if not re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", tag):
        raise ValueError(f"Expected a stable vX.Y.Z release tag, got {tag!r}")
    return tag[1:]


def resolve_release(tag: str) -> str:
    release_version(tag)
    if os.environ.get("GITHUB_REPOSITORY") != "mlflow/mlflow":
        raise ValueError("Charts can only be published from mlflow/mlflow")
    release = json.loads(run("gh", "api", f"repos/mlflow/mlflow/releases/tags/{tag}"))
    if release["draft"] or release["prerelease"] or not release["published_at"]:
        raise ValueError(f"{tag} is not a published stable release")
    commit: str = json.loads(run("gh", "api", f"repos/mlflow/mlflow/commits/{tag}"))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"Invalid release commit: {commit!r}")
    return commit


def chart_digest(version: str) -> str | None:
    # An anonymous read also verifies that the chart is publicly available.
    token_url = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{CHART_REPOSITORY}:pull"
    with urllib.request.urlopen(token_url, timeout=30) as response:
        token = json.load(response)["token"]
    request = urllib.request.Request(
        f"https://ghcr.io/v2/{CHART_REPOSITORY}/manifests/{version}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.manifest.v1+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            digest: str | None = response.headers["Docker-Content-Digest"]
    except urllib.error.HTTPError as error:
        # Never turn permission, authentication, or network failures into a push.
        if error.code == 404:
            errors = json.load(error).get("errors", [])
            if errors and all(e.get("code") == "MANIFEST_UNKNOWN" for e in errors):
                return None
        raise
    if not digest or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"Registry returned an invalid chart digest: {digest!r}")
    return digest


def chart_contents(package: Path) -> dict[str, bytes]:
    # Compare payloads rather than gzip/tar timestamps; do not extract registry files.
    with tarfile.open(package) as archive:
        result = {}
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.name in result:
                raise ValueError(f"Unexpected chart archive member: {member.name}")
            content = archive.extractfile(member)
            assert content is not None
            result[member.name] = content.read()
        return result


def verify_chart(package: Path, version: str) -> None:
    metadata = yaml.safe_load(run("helm", "show", "chart", str(package)))
    for key, expected in (("name", "mlflow"), ("version", version), ("appVersion", version)):
        if metadata.get(key) != expected:
            raise ValueError(f"Expected chart {key}={expected}, got {metadata.get(key)!r}")
    rendered = run(
        "helm",
        "template",
        "mlflow",
        str(package),
        "--set",
        "mlflow.backendStoreUri=sqlite:////tmp/mlflow.db",
        "--set",
        "garbageCollection.enabled=true",
    )
    images = {}
    for resource in yaml.safe_load_all(rendered):
        if resource and resource["kind"] in ("Deployment", "CronJob"):
            spec = resource["spec"]
            if resource["kind"] == "CronJob":
                spec = spec["jobTemplate"]["spec"]
            images[resource["kind"]] = [
                container["image"] for container in spec["template"]["spec"]["containers"]
            ]
    expected_image = f"ghcr.io/mlflow/mlflow:v{version}-full"
    if images != {"Deployment": [expected_image], "CronJob": [expected_image]}:
        raise ValueError(f"Chart workloads must use {expected_image}: {images}")


def pull_chart(version: str, destination: Path) -> Path:
    run("helm", "pull", CHART_URI, "--version", version, "--destination", str(destination))
    return destination / f"mlflow-{version}.tgz"


def publish_chart(tag: str, chart_dir: Path, source_sha: str, dry_run: bool = False) -> None:
    version = release_version(tag)
    if resolve_release(tag) != source_sha:
        raise ValueError("The release tag changed after checkout")
    if not dry_run:
        run("docker", "manifest", "inspect", f"ghcr.io/mlflow/mlflow:{tag}-full")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        run(
            "helm",
            "package",
            str(chart_dir),
            "--version",
            version,
            "--app-version",
            version,
            "--destination",
            temporary,
        )
        package = directory / f"mlflow-{version}.tgz"
        verify_chart(package, version)
        if dry_run:
            summary = (
                "Dry run succeeded; no GHCR registry or image operations were performed.\n\n"
                f"- Source commit: `{source_sha}`\n"
                f"- Chart version / appVersion: `{version}`\n"
                f"- Expected image: `ghcr.io/mlflow/mlflow:{tag}-full`\n"
                "- Local package metadata, Deployment and CronJob images verified.\n"
            )
            print(summary)
            if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
                with open(summary_path, "a") as stream:
                    stream.write(summary)
            return
        expected = chart_contents(package)
        pulled = directory / "pulled"
        pulled.mkdir()
        digest = chart_digest(version)
        if digest is not None:
            existing = pull_chart(version, pulled)
            if chart_contents(existing) != expected:
                raise ValueError(
                    f"Chart {version} already exists with different content; refusing overwrite"
                )
            result = "Already published (identical content)"
        else:
            run("helm", "push", str(package), "oci://ghcr.io/mlflow/charts")
            result = "Published"
        verified = pull_chart(version, pulled)
        verify_chart(verified, version)
        if chart_contents(verified) != expected:
            raise ValueError("Published chart content does not match the release package")
        verified_digest = chart_digest(version)
        if verified_digest is None or (digest is not None and verified_digest != digest):
            raise ValueError("Chart digest disappeared or changed during verification")
        summary = (
            f"{result}\n\n"
            f"- Source commit: `{source_sha}`\n"
            f"- Chart: `{CHART_URI}:{version}`\n"
            f"- Chart version / appVersion: `{version}`\n"
            f"- Image: `ghcr.io/mlflow/mlflow:{tag}-full`\n"
            f"- OCI digest: `{verified_digest}`\n"
            "- Pulled content, metadata, Deployment and CronJob images verified.\n"
        )
        print(summary)
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary_path, "a") as stream:
                stream.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("release_tag")
    publish = subparsers.add_parser("publish")
    publish.add_argument("release_tag")
    publish.add_argument("--chart-dir", type=Path, required=True)
    publish.add_argument("--source-sha", required=True)
    publish.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "resolve":
        print(f"commit={resolve_release(args.release_tag)}")
    else:
        publish_chart(args.release_tag, args.chart_dir, args.source_sha, args.dry_run)
