# pypi

Minimal type-safe client for [PyPI's JSON API](https://warehouse.pypa.io/api-reference/json.html), used by scripts in `dev/`.

```python
from pypi import get_package

pkg = get_package("requests")
pkg.latest_version  # Version("2.32.3")
pkg.releases[-1].upload_time  # datetime (UTC)
pkg.releases[-1].yanked  # bool
pkg.releases[-1].requires_python  # SpecifierSet | None
```
