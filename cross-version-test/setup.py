from distutils.core import setup
import importlib


def read_version() -> str:
    return importlib.import_module("cross_version_test.version").version


setup(
    name="cross-version-test",
    version=read_version(),
    description="CLI to run cross-version tests",
    author="Databricks",
    install_requires=[
        "click",
        "Jinja2",
        "pyyaml",
        "pydantic",
        "packaging",
        "requests",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "types-requests",
            "types-PyYAML",
            "mypy",
        ]
    },
    entry_points={
        "console_scripts": [
            "cross-version-test=cross_version_test.cli:cli",
        ],
    },
)
