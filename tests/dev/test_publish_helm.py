import io
import json
import shutil
import subprocess
import tarfile
import urllib.error
from unittest.mock import ANY, Mock, call

import pytest
import yaml

from dev import publish_helm


@pytest.mark.parametrize(
    "tag",
    [
        "v3.16.0rc1",
        "3.16.0",
        "v03.16.0",
        "v3.16",
        "v3.16.0+build",
        "model-catalog/latest",
        "v3.16.0\n",
    ],
)
def test_reject_invalid_release_tag(tag):
    with pytest.raises(ValueError, match="stable vX.Y.Z"):
        publish_helm.release_version(tag)


@pytest.mark.parametrize("changes", [{"draft": True}, {"prerelease": True}, {"published_at": None}])
def test_reject_unpublished_release(monkeypatch, changes):
    monkeypatch.setenv("GITHUB_REPOSITORY", "mlflow/mlflow")
    release = {"draft": False, "prerelease": False, "published_at": "2026-09-01", **changes}
    monkeypatch.setattr(publish_helm, "run", Mock(return_value=json.dumps(release)))
    with pytest.raises(ValueError, match="published stable release"):
        publish_helm.resolve_release("v3.16.0")


def test_reject_fork(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/mlflow")
    with pytest.raises(ValueError, match="only be published from"):
        publish_helm.resolve_release("v3.16.0")


@pytest.mark.parametrize("field", ["name", "version", "appVersion"])
def test_verify_rejects_wrong_chart_metadata(monkeypatch, field):
    metadata = {"name": "mlflow", "version": "3.16.0", "appVersion": "3.16.0"}
    metadata[field] = "unexpected"
    monkeypatch.setattr(publish_helm, "run", Mock(return_value=yaml.safe_dump(metadata)))
    with pytest.raises(ValueError, match=f"Expected chart {field}"):
        publish_helm.verify_chart("chart.tgz", "3.16.0")


@pytest.mark.parametrize("kind", ["Deployment", "CronJob"])
def test_verify_rejects_stale_workload_image(monkeypatch, kind):
    documents = []
    for workload in ("Deployment", "CronJob"):
        version = "3.15.2" if workload == kind else "3.16.0"
        spec = {
            "template": {
                "spec": {"containers": [{"image": f"ghcr.io/mlflow/mlflow:v{version}-full"}]}
            }
        }
        if workload == "CronJob":
            spec = {"jobTemplate": {"spec": spec}}
        documents.append({"kind": workload, "spec": spec})
    monkeypatch.setattr(
        publish_helm,
        "run",
        Mock(
            side_effect=[
                yaml.safe_dump({"name": "mlflow", "version": "3.16.0", "appVersion": "3.16.0"}),
                yaml.safe_dump_all(documents),
            ]
        ),
    )
    with pytest.raises(ValueError, match="Chart workloads must use"):
        publish_helm.verify_chart("chart.tgz", "3.16.0")


def test_changed_release_tag_stops_before_registry_access(monkeypatch, tmp_path):
    monkeypatch.setattr(publish_helm, "resolve_release", Mock(return_value="b" * 40))
    command = Mock()
    monkeypatch.setattr(publish_helm, "run", command)
    with pytest.raises(ValueError, match="release tag changed"):
        publish_helm.publish_chart("v3.16.0", tmp_path, "a" * 40)
    command.assert_not_called()


@pytest.mark.parametrize(
    ("status", "code", "absent"),
    [
        (404, "MANIFEST_UNKNOWN", True),
        (404, "DENIED", False),
        (401, "UNAUTHORIZED", False),
        (403, "DENIED", False),
        (500, "UNKNOWN", False),
    ],
)
def test_registry_errors_fail_closed(monkeypatch, status, code, absent):
    error = urllib.error.HTTPError(
        "https://ghcr.io/test",
        status,
        "registry error",
        {},
        io.BytesIO(json.dumps({"errors": [{"code": code}]}).encode()),
    )
    monkeypatch.setattr(
        publish_helm.urllib.request,
        "urlopen",
        Mock(side_effect=[io.BytesIO(b'{"token":"test"}'), error]),
    )
    if absent:
        assert publish_helm.chart_digest("3.16.0") is None
    else:
        with pytest.raises(urllib.error.HTTPError, match="registry error"):
            publish_helm.chart_digest("3.16.0")


def test_chart_comparison_ignores_archive_timestamps(tmp_path):
    for timestamp in (1, 2):
        with tarfile.open(tmp_path / f"{timestamp}.tgz", "w:gz") as archive:
            info = tarfile.TarInfo("mlflow/Chart.yaml")
            info.size = 3
            info.mtime = timestamp
            archive.addfile(info, io.BytesIO(b"abc"))
    assert publish_helm.chart_contents(tmp_path / "1.tgz") == publish_helm.chart_contents(
        tmp_path / "2.tgz"
    )


@pytest.mark.parametrize(
    "state", ["absent", "identical", "conflicting", "registry-error", "missing-image"]
)
def test_publication_retries(monkeypatch, tmp_path, state):
    monkeypatch.setattr(publish_helm, "resolve_release", Mock(return_value="a" * 40))
    commands = []

    def run(*args):
        commands.append(args)
        if state == "missing-image" and args[:2] == ("docker", "manifest"):
            raise subprocess.CalledProcessError(1, args)
        return ""

    monkeypatch.setattr(publish_helm, "run", run)
    monkeypatch.setattr(publish_helm, "verify_chart", Mock())
    monkeypatch.setattr(publish_helm, "pull_chart", Mock(return_value=tmp_path / "existing.tgz"))
    monkeypatch.setattr(
        publish_helm,
        "chart_contents",
        Mock(
            side_effect=[
                {"chart": b"candidate"},
                {"chart": b"different" if state == "conflicting" else b"candidate"},
                {"chart": b"candidate"},
            ]
        ),
    )
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        publish_helm,
        "chart_digest",
        Mock(
            side_effect=RuntimeError("Registry unavailable")
            if state == "registry-error"
            else [None if state == "absent" else digest, digest]
        ),
    )
    if state in ("conflicting", "registry-error", "missing-image"):
        with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
            publish_helm.publish_chart("v3.16.0", tmp_path, "a" * 40)
    else:
        publish_helm.publish_chart("v3.16.0", tmp_path, "a" * 40)
    assert any(command[:2] == ("helm", "push") for command in commands) == (state == "absent")
    assert (
        publish_helm.verify_chart.call_count
        == {
            "absent": 2,
            "identical": 2,
            "conflicting": 1,
            "registry-error": 1,
            "missing-image": 0,
        }[state]
    )
    assert all(call.args[1] == "3.16.0" for call in publish_helm.verify_chart.call_args_list)


def test_dry_run_never_accesses_registry_or_image(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(publish_helm, "resolve_release", Mock(return_value="a" * 40))
    command = Mock(return_value="")
    verify = Mock()
    digest = Mock()
    pull = Mock()
    contents = Mock()
    monkeypatch.setattr(publish_helm, "run", command)
    monkeypatch.setattr(publish_helm, "verify_chart", verify)
    monkeypatch.setattr(publish_helm, "chart_digest", digest)
    monkeypatch.setattr(publish_helm, "pull_chart", pull)
    monkeypatch.setattr(publish_helm, "chart_contents", contents)

    publish_helm.publish_chart("v3.16.0", tmp_path, "a" * 40, dry_run=True)

    assert "Dry run succeeded" in capsys.readouterr().out
    verify.assert_called_once()
    assert verify.call_args.args[1] == "3.16.0"
    digest.assert_not_called()
    pull.assert_not_called()
    contents.assert_not_called()
    assert command.call_args_list == [
        call(
            "helm",
            "package",
            str(tmp_path),
            "--version",
            "3.16.0",
            "--app-version",
            "3.16.0",
            "--destination",
            ANY,
        )
    ]


@pytest.mark.skipif(
    shutil.which("helm") is None, reason="Helm is required for packaging regression"
)
def test_package_stale_source_without_modifying_it(tmp_path):
    chart = tmp_path / "charts"
    shutil.copytree("charts", chart)
    metadata = chart / "Chart.yaml"
    stale = yaml.safe_load(metadata.read_text())
    stale.update(version="0.1.1", appVersion="3.15.2")
    metadata.write_text(yaml.safe_dump(stale))
    before = {
        path.relative_to(chart): path.read_bytes() for path in chart.rglob("*") if path.is_file()
    }
    publish_helm.run(
        "helm",
        "package",
        str(chart),
        "--version",
        "3.16.0",
        "--app-version",
        "3.16.0",
        "--destination",
        str(tmp_path),
    )
    publish_helm.verify_chart(tmp_path / "mlflow-3.16.0.tgz", "3.16.0")
    assert {
        path.relative_to(chart): path.read_bytes() for path in chart.rglob("*") if path.is_file()
    } == before
