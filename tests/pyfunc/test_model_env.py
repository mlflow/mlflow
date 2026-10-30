import json
import os
import posixpath
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest

from mlflow.pyfunc import _prebuild_env_internal
from mlflow.pyfunc.dbconnect_artifact_cache import archive_directory, extract_archive_to_dir
from mlflow.utils.os import is_windows
from mlflow.utils.virtualenv import _normalize_env_symlinks

pytestmark = pytest.mark.skipif(
    is_windows(), reason="Databricks environment archives require POSIX"
)


def _assert_safe_symlinks(archive_path):
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.issym():
                assert not posixpath.isabs(member.linkname)
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(member.name), member.linkname)
                )
                assert target != ".."
                assert not target.startswith("../")


@pytest.mark.parametrize("env_manager", ["virtualenv", "uv"])
@pytest.mark.parametrize("bundled_python", [True, False])
def test_prebuild_env_normalizes_python_symlinks(
    tmp_path, monkeypatch, env_manager, bundled_python
):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    monkeypatch.setattr("mlflow.pyfunc._PREBUILD_ENV_ROOT_LOCATION", str(build_dir))
    env_root = build_dir / "mlflow-test"
    python_relative_path = Path("virtualenv_envs/mlflow-test/bin/python")
    python_contents = b"bundled interpreter"

    def prepare_env(**kwargs):
        python_target = (
            env_root / "pyenv_root/versions/3.12.3/bin/python"
            if bundled_python
            else tmp_path / "system/bin/python3.12"
        )
        python_target.parent.mkdir(parents=True)
        python_target.write_bytes(python_contents)
        python_target.chmod(0o755)
        python_path = env_root / python_relative_path
        python_path.parent.mkdir(parents=True)
        python_path.symlink_to(python_target)
        (python_path.parent / "python3").symlink_to("python")
        cache_path = env_root / "pip_cache_pkgs"
        cache_path.mkdir()
        (cache_path / "cached-wheel").write_text("excluded")

    with mock.patch("mlflow.pyfunc.get_flavor_backend") as get_backend:
        get_backend.return_value.prepare_env.side_effect = prepare_env
        archive_path = _prebuild_env_internal("model", "mlflow-test", str(tmp_path), env_manager)

    assert not env_root.exists()
    if not bundled_python:
        shutil.rmtree(tmp_path / "system")
    _assert_safe_symlinks(archive_path)
    restored_dir = tmp_path / "restored"
    extract_archive_to_dir(archive_path, restored_dir)
    python_path = restored_dir / python_relative_path
    assert python_path.read_bytes() == python_contents
    assert os.access(python_path, os.X_OK)
    assert python_path.is_symlink() == bundled_python
    assert (python_path.parent / "python3").read_bytes() == python_contents
    assert not (restored_dir / "pip_cache_pkgs").exists()


def test_normalize_env_directory_symlinks(tmp_path):
    env_root = tmp_path / "env"
    internal_dir = env_root / "lib"
    internal_dir.mkdir(parents=True)
    (internal_dir / "internal.py").write_text("internal")
    (env_root / "lib64").symlink_to(internal_dir, target_is_directory=True)
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "external.py").write_text("external")
    (external_dir / "alias.py").symlink_to(external_dir / "external.py")
    (env_root / "include").symlink_to(external_dir, target_is_directory=True)

    _normalize_env_symlinks(env_root)
    archive_path = archive_directory(env_root, tmp_path / "env.tar.gz")
    shutil.rmtree(env_root)
    shutil.rmtree(external_dir)
    _assert_safe_symlinks(archive_path)
    restored_dir = tmp_path / "restored"
    extract_archive_to_dir(archive_path, restored_dir)
    assert (restored_dir / "lib64/internal.py").read_text() == "internal"
    assert (restored_dir / "include/external.py").read_text() == "external"
    assert (restored_dir / "include/alias.py").read_text() == "external"
    assert not (restored_dir / "include").is_symlink()
    assert not (restored_dir / "include/alias.py").is_symlink()


@pytest.mark.skipif(sys.platform != "linux", reason="Databricks prebuilt environments run on Linux")
@pytest.mark.parametrize("env_manager", ["virtualenv", "uv"])
@pytest.mark.parametrize("bundled_python", [True, False])
def test_prebuilt_env_python_runs(tmp_path, monkeypatch, env_manager, bundled_python):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    monkeypatch.setattr("mlflow.pyfunc._PREBUILD_ENV_ROOT_LOCATION", str(build_dir))
    env_root = build_dir / "mlflow-test"
    env_dir = env_root / "virtualenv_envs/mlflow-test"

    def prepare_env(**kwargs):
        if env_manager == "virtualenv":
            subprocess.check_call([sys.executable, "-m", "venv", "--without-pip", str(env_dir)])
        else:
            subprocess.check_call([
                "uv",
                "venv",
                "--python",
                sys.executable,
                "--no-project",
                str(env_dir),
            ])
        python_path = env_dir / "bin/python"
        assert python_path.is_symlink()
        if bundled_python:
            python_target = env_root / "pyenv_root/versions/test/bin/python"
            python_target.parent.mkdir(parents=True)
            shutil.copy2(python_path.resolve(), python_target)
            python_path.unlink()
            python_path.symlink_to(python_target)
        assert python_path.resolve().is_relative_to(env_root) == bundled_python
        site_packages = subprocess.check_output(
            [str(python_path), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
        Path(site_packages, "model_dependency.py").write_text("def predict(x): return x * 2\n")

    with mock.patch("mlflow.pyfunc.get_flavor_backend") as get_backend:
        get_backend.return_value.prepare_env.side_effect = prepare_env
        archive_path = _prebuild_env_internal("model", "mlflow-test", str(tmp_path), env_manager)

    _assert_safe_symlinks(archive_path)
    restored_dir = tmp_path / "restored"
    extract_archive_to_dir(archive_path, restored_dir)
    python_path = restored_dir / "virtualenv_envs/mlflow-test/bin/python"
    assert python_path.is_symlink() == bundled_python
    output = subprocess.check_output(
        [
            str(python_path),
            "-c",
            "import json, ssl, sqlite3, sys; import model_dependency; "
            "print(json.dumps({'prediction': model_dependency.predict(21), 'prefix': sys.prefix}))",
        ],
        text=True,
    )
    result = json.loads(output)
    assert result["prediction"] == 42
    assert Path(result["prefix"]) == python_path.parent.parent
