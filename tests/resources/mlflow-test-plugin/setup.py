import warnings

from setuptools import setup

warnings.warn(
    "setup.py is deprecated. Please use pyproject.toml for package configuration.",
    FutureWarning,
    stacklevel=2,
)

# Configuration is now handled by pyproject.toml
# This setup.py is maintained for backward compatibility only
setup()
