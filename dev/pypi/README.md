# pypi

A minimal, type-safe client for fetching package metadata from
[PyPI's JSON API](https://warehouse.pypa.io/api-reference/json.html).

Built for use by scripts under `dev/` (e.g. `update_requirements.py`,
`update_ml_package_versions.py`, `show_package_release_dates.py`) that
currently re-implement PyPI fetching ad-hoc.

## Features

- Versions are auto-coerced to `packaging.version.Version` (invalid versions
  like `pytz==2004d` are silently dropped).
- Transient network errors and `408/425/429/5xx` responses are retried with
  exponential backoff. `404` and other 4xx errors fail fast.
- Responses are cached in-process via `functools.cache` (one fetch per
  package per interpreter).
- `PYPI_URL` environment variable overrides the base URL (for proxies/mirrors).
- Stdlib only, plus `packaging`.

## Usage

```python
from pypi import get_package

pkg = get_package("requests")

pkg.name  # "requests"
pkg.latest_version  # Version("2.32.3")  — newest non-pre, non-dev, non-yanked
pkg.versions  # tuple[Version, ...]

release = pkg.get_release("2.32.3")
release.upload_time  # datetime | None  (earliest distribution upload)
release.yanked  # bool
```

## API

- `get_package(name) -> Package`
- `clear_cache() -> None`
- `PyPIError` — raised on missing packages or exhausted retries
- `Package`, `Release` — frozen dataclasses

## Install (editable, for local dev)

```bash
uv pip install -e dev/pypi
```
