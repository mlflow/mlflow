# https://setuptools.pypa.io/en/latest/build_meta.html#dynamic-build-dependencies-and-other-build-meta-tweaks
import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as _orig_build_meta
from setuptools.build_meta import *  # noqa: F403


def get_requires_for_build_sdist(config_settings=None):
    try:
        sys.stdout.write("Updating SHA in mlflow/sha.py...\n")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        Path("mlflow", "sha.py").write_text(f'sha = "{sha}"\n')
    except Exception as e:
        sys.stderr.write(f"Failed to write SHA to mlflow/sha.py: {e}\n")
    return _orig_build_meta.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_wheel(config_settings=None):
    try:
        sys.stdout.write("Updating SHA in mlflow/sha.py...\n")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        Path("mlflow", "sha.py").write_text(f'sha = "{sha}"\n')
    except Exception as e:
        sys.stderr.write(f"Failed to write SHA to mlflow/sha.py: {e}\n")
    return _orig_build_meta.get_requires_for_build_wheel(config_settings)
