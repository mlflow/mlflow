# pypi

Minimal type-safe client for [PyPI's JSON API](https://warehouse.pypa.io/api-reference/json.html), used by scripts in `dev/`.

```python
from pypi import get_package, aget_packages

pkg = get_package("requests")
pkg.latest_version  # Version("2.32.3")
pkg.releases[-1].upload_time  # datetime (UTC)
pkg.releases[-1].yanked  # bool
pkg.releases[-1].requires_python  # SpecifierSet | None
```

- Stdlib + `packaging` only.
- Versions → `packaging.Version`; `requires_python` → `SpecifierSet`.
- In-process cache (`@functools.cache`); `clear_cache()` to reset.
- Retries `URLError`/`5xx`/`429` with exponential backoff; `404` fails fast.
- `aget_packages(names)` for concurrent batch fetch.
- `PYPI_URL` env var overrides the base URL.
