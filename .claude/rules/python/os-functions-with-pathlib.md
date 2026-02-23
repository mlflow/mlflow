---
paths: "**/*.py"
---

# OS Functions with Pathlib

When you have a `pathlib.Path` object, use its built-in methods instead of `os` module functions. This is more readable, type-safe, and follows object-oriented principles.

```python
from pathlib import Path

path = Path("some/file.txt")

# Bad
import os

os.path.exists(path)
os.remove(path)

# Good
path.exists()
path.unlink()
```
