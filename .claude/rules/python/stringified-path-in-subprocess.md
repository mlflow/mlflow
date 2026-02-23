---
paths: "**/*.py"
---

# Stringified Path in Subprocess

Avoid converting `pathlib.Path` objects to strings when passing them to `subprocess` functions. Modern Python (3.8+) accepts Path objects directly, making the code cleaner and more type-safe.

```python
import subprocess
from pathlib import Path

path = Path("some/script.py")

# Bad
subprocess.check_call(["foo", "bar", str(path)])

# Good
subprocess.check_call(["foo", "bar", path])
```
